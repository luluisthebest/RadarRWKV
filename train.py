import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_DEBUG'] = 'INFO'
import json
import argparse
import torch
import random
import numpy as np
import yaml
import glob
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.MVIBRadarRWKV_dp import SparseRadarRWKV, HierarchicalMIBCriterion
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
from dataset.rad_cube_loader import RADCUBE_DATASET
from dataset.data_preparation import radarcube_lidarcube_loss
from tqdm import tqdm
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from utils.distributed_utils import *
from utils.eval_vib import run_evaluation, weighted_mean
from utils.compute_metrics import hoyer_metric
from torch.utils.data import DataLoader
import bisect
import math
import segmentation_models_pytorch as smp
                            
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main(config):

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='4D radar cube Training')
    parser.add_argument('-c', '--config', default='./configs/RaDelft.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--dist-url', default='env://', help='Url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    parser.add_argument('--zero_init_residual', default="true", action='store_true', help='zero init all residual blocks')
    parser.add_argument('--device', default = "cuda")
    parser.add_argument('--num_frames', default=1, help='number of consecutive frames')

    args = parser.parse_args()            
    config = json.load(open(args.config))
    
    # other configurations
    cos = False       # learn schedule
    dop = False      # if use doppler for Radial
    fp16 = False
    ada_loss = False  # if adaptive loss for Radial
    wd = 1e-3
    Greverse = False
    dataset = 'radelft'
    multiclass = False    # if multiclass or bianry class
    schedule = 'plateau'     # step or plateau
    lambda_kld = 1e-2
    lambda_klg = 1   

    # create experience name
    curr_date = datetime.now()
    exp_name = dataset + '_' + config['name'] + '_' + curr_date.strftime('%b-%d-%Y_%H:%M:%S')
    # exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y')
    print(exp_name)

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name ).mkdir(parents=True, exist_ok=True)

    # copy the config file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)
    
    if dataset == 'radelft':
        range_axis = np.arange(config['range_cell_size'], config['max_range'] + config['range_cell_size'], config['range_cell_size'])
        range_axis = range_axis[10:-3]
        config['range_axis'] = range_axis

        wx_vec = np.linspace(-np.pi, np.pi, config['angle_fft_size'])
        wx_vec = wx_vec[8:248]
        azimuth_axis = np.arcsin(wx_vec / (2 * np.pi * 0.4972))
        config['azimuth_axis'] = azimuth_axis

        wz_vec = np.linspace(-np.pi, np.pi, config['ele_fft_size'])
        wz_vec = wz_vec[47:81]
        elevation_axis = np.arcsin(wz_vec / (2 * np.pi * 0.4972))
        config['elevation_axis'] = elevation_axis
    
    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # 分布式训练
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    scaler = torch.amp.GradScaler(enabled=fp16)

    # Load the datasetset
    if dataset == 'radial':
        enc = ra_encoder(geometry = config['dataset']['geometry'], 
                            statistics = config['dataset']['statistics'],
                            regression_layer = 3 if dop else 2)
        
        radial_dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=True,perform_FFT=config['data_mode'])

        train_loader, train_sampler, val_loader, test_loader = CreateDataLoaders(radial_dataset, config['dataloader'],config['seed'], args.distributed)
    elif dataset == 'radelft':
        train_dataset = RADCUBE_DATASET(mode='train', params=config, multiclss=multiclass)
        val_dataset = RADCUBE_DATASET(mode='val', params=config, multiclss=multiclass)
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, 
                                sampler=train_sampler,
                                batch_size=config['train']['batch_size'], 
                                num_workers=config['train']['num_workers'],
                                pin_memory=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset)
        val_loader =  DataLoader(val_dataset, 
                                sampler=val_sampler,
                                batch_size=config['val']['batch_size'], 
                                num_workers=config['val']['num_workers'],
                                pin_memory=True)

    print("Loading model and tokenizer...")
    # Create the model
    net = SparseRadarRWKV(device=device)
    net.eval()
    net.to(device)
    # if get_rank() == 0:
    print('Param count: ', sum(m.numel() for m in net.parameters()))
    for param in net.parameters():
        param.data = param.data.contiguous()
    if dist.is_initialized():
        dist.broadcast(param.data, src=0)

    if args.distributed:# and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    # Optimizer and schedular
    if schedule == 'step':
        lr = float(config['optimizer']['lr'])
        step_size = int(config['lr_scheduler']['step_size'])
        gamma = float(config['lr_scheduler']['gamma'])
            
        if cos:
            if ada_loss:
                optimizer = optim.AdamW(
                    [
                        {'params': filter(lambda p: p.requires_grad, net.parameters())},
                        {'params': adaptive_loss.parameters()}
                    ], lr=2e-4, weight_decay=wd)
            else:
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, weight_decay=wd)  #5e-4)
            warmup_epoch = 5
            scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epoch)
            scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs']-warmup_epoch, eta_min=1e-5)
            scheduler = lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_epoch])
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif schedule == 'plateau':
        lr = float(config['optimizer']['lr'])
        param_group = [
            {
                'params': [p for n, p in net.named_parameters() 
                        if 'vib_bottleneck' in n and p.requires_grad],
                'lr': lr*0.1,
            },
            {
                'params': [p for n, p in net.named_parameters() 
                        if not any(x in n for x in ['vib']) and p.requires_grad],
                'lr': lr,
            }]
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer = optim.Adam(param_group)
        # optimizer = optim.RAdam(param_group)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, min_lr=1e-6)              
    
    # 分布式计算总步数
    max_steps = math.ceil(len(train_dataset) / (config['train']['batch_size'] * dist.get_world_size())) * config['num_epochs']
    kl_annealing_scheduler = kl_annealing(
                annealing_value_start=0,
                annealing_value_end=1,
                wait_before_warmup=max_steps * 0.333,  # wait for 30% of the steps
                end_of_warmup=max_steps * 0.667,  # warmup
                type='linear',)
   
    num_epochs=int(config['num_epochs'])
    # for idx, (name, param) in enumerate(net.named_parameters()):
    #     print(f"Index {idx}: {name}")
    # all_params = list(net.parameters())
    # param_254 = all_params[254]
    # print(f"Parameter 254 details:")
    # Shape=param_254.shape
    # Stride=param_254.stride()
    # Device=param_254.device
    # Dtype=param_254.dtype
    # contiguous=param_254.is_contiguous()

    if args.distributed:
        # net = nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])        

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    iteration = 0
    #global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'Pd':[],'Pfa':[]}
    best_mAP = 0

    # Setup random seed
    torch.manual_seed(config['seed']+get_rank())
    np.random.seed(config['seed']+get_rank())
    random.seed(config['seed']+get_rank())
    torch.cuda.manual_seed(config['seed']+get_rank())

    if args.resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(args.resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        iteration = dict['global_step']

        print('       ... Start at epoch:',startEpoch)

    torch.autograd.set_detect_anomaly(True)
    #==========监测梯度===========#
    # monitor(net, writer, -1)

    for epoch in range(startEpoch,num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        
        ###################
        ## Training loop ##
        ###################
            
        net.train()
        running_loss = 0.0

        # shuffle
        if args.distributed:
            train_sampler.set_epoch(epoch)        
        
        for i, data in enumerate(train_loader):
        
            # dataset input
            inputs = data[0].to(device)   
            label_map = data[1].to(device).float()
            
            # reset the gradient
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=fp16):
                # with torch.no_grad():
                    outputs, kl_g_list, kl_d_list, alpha_list = net(inputs)
            
              alpha_list_detached = [alpha.detach() for alpha in alpha_list]
              classif_loss = radarcube_lidarcube_loss(outputs, label_map, config, multiclass=multiclass)
              kl_factor = kl_annealing_scheduler(iteration)
              klg = (weighted_mean(kl_g_list, True) * lambda_klg) * kl_factor
              kld = (weighted_mean(kl_d_list, True) * lambda_kld) * kl_factor

              loss = classif_loss + klg + kld          
            
            loss_reduced = reduce_value(loss)
            cls_reduced = reduce_value(classif_loss)
            # klg_reduced = reduce_value(klg)
            # kld_reduced = reduce_value(kld)
        
            # backprop
            with torch.autograd.set_detect_anomaly(True):
                try:
                    scaler.scale(loss).backward()
                except RuntimeError as e:
                    print(f"Anomaly detected: {e}")
            # scale_loss = scaler.scale(loss)  # for debug
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if get_rank() == 0:
                writer.add_scalar('Loss/train', loss_reduced.item(), iteration)
                writer.add_scalar('Loss/train_cls', cls_reduced.item(), iteration)
                writer.add_scalar('Loss/train_klg', (weighted_mean(kl_g_list, True) * lambda_klg).item(), iteration)
                writer.add_scalar('Loss/train_kld', (weighted_mean(kl_d_list, True) * lambda_kld).item(), iteration)
                writer.add_scalar('parameters/learning_rate_vib', scheduler.get_last_lr()[0], iteration)
                writer.add_scalar('parameters/learning_rate_other', scheduler.get_last_lr()[1], iteration)
                writer.add_scalar('parameters/kl_factor', kl_factor, iteration)
                writer.add_scalar('alpha/layer_3_min', alpha_list_detached[0].min(), iteration)
                writer.add_scalar('alpha/layer_3_max', alpha_list_detached[0].max(), iteration)
     
                # statistics
                running_loss += loss_reduced.item() * inputs.size(0)
            
                kbar.update(i, values=[("loss", loss_reduced.item())])
            
            iteration += 1
            # functional.reset_net(net)
            # break   # for debug
        torch.distributed.barrier()
        if schedule == 'plateau':
            scheduler.step(loss_reduced)
        else:
            scheduler.step()
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        #==========监测梯度===========#
        # monitor(net, writer, epoch)
                
        ######################
        ## validation phase ##
        ######################
        eval = run_evaluation(config['name'], epoch, net, val_loader, encoder=None, check_perf=(epoch>=0), T=args.num_frames, 
                              loss_ce=None, loss_fl=None, detection_loss=radarcube_lidarcube_loss,segmentation_loss=None,
                              losses_params=None, doppler=dop, config=config, adaptive_loss=adaptive_loss if ada_loss else None,
                              kl_factor=kl_factor, lambda_klg=lambda_klg, lambda_kld=lambda_kld,)

        history['val_loss'].append(eval['loss'])
        history['Pd'].append(eval['Pd'])
        history['Pfa'].append(eval['Pfa'])

        kbar.add(1, values=[("val_loss", eval['loss']),("Pd", eval['Pd']),("Pfa", eval['Pfa'])])

        if get_rank() == 0:
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Loss/val', eval['loss'], epoch)
            writer.add_scalar('Loss/val_cls', eval['loss_cls'], epoch)
            writer.add_scalar('Loss/val_klg', eval['klg'], epoch)
            writer.add_scalar('Loss/val_kld', eval['kld'], epoch)
            writer.add_scalar('Metrics/Pd', eval['Pd'], epoch)
            writer.add_scalar('Metrics/Pfa', eval['Pfa'], epoch)

        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        if get_rank() == 0:
            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, eval['loss'],eval['Pd'],eval['Pfa'])
            filename = output_folder / exp_name / name_output_file

            checkpoint={}
            checkpoint['net_state_dict'] = net.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['history'] = history
            checkpoint['global_step'] = iteration

            torch.save(checkpoint,filename)
        
        print('')



def monitor(net, writer, epoch):
    total_norm = 0
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Epoch: {epoch}, Gradient Norm: {total_norm:.4f}")
    writer.add_scalar('Gradient Norm', total_norm, epoch)
    for name, param in net.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"{name}", param, epoch) #参数直方图
            if param.grad is not None:
                writer.add_histogram(f"{name}/grad", param.grad, epoch)  #梯度直方图

        param_data = param.detach().cpu().numpy().flatten()
        if np.max(np.abs(param_data)) < 1e-5:
            print(name,"的参数全零")
        elif np.std(param_data) < 0.01:
            print(name,"的参数分布过窄: ", np.std(param_data))
        
        # 梯度分析
        if param.grad is not None:
            grad_data = param.grad.detach().cpu().numpy().flatten()
            if np.max(np.abs(grad_data)) < 1e-6:
                print(name,"的梯度消失")
            elif np.max(np.abs(grad_data)) > 1000:
                print(name,"的梯度爆炸")
            
            # 梯度参数比例
            ratio = np.abs(grad_data) / (np.abs(param_data) + 1e-7)
            if np.median(ratio) > 0.5:
                print(name,"的梯度参数比例过高")


def lr_foo(epoch):    
    lr_rate = [0.05, 0.1, 0.2]   
    lr_scheduler_epoch = [10, 20, 30]
    if epoch < 3:
        # warm up lr
        lr_scale = lr_rate[epoch]
        # lr_scale = 0.4 ** (3 - epoch)
    else:
        # warmup schedule
        lr_pos = int(-1 - bisect.bisect_left(lr_scheduler_epoch, epoch))
        if lr_pos < -3:
            lr_scale = max(lr_rate[0] * (0.98 ** epoch), 0.03 )
        else:
            lr_scale = lr_rate[lr_pos]
        # lr_scale = self.config.lr_rate[-1] ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch) + 1)
        # lr_scale = 0.4 ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch) + 2)
        # 0.3 * (0.98 ** epoch) 
        # # 0.4 ** 
    return lr_scale


class kl_annealing:
    def __init__(
        self,
        end_of_warmup,
        wait_before_warmup=0,
        annealing_value_start=0,
        annealing_value_end=1,
        type="linear",
    ):
        self.annealing_value_start = annealing_value_start
        self.annealing_value_end = annealing_value_end
        self.end_of_warmup = end_of_warmup
        self.type = type
        self.wait_before_warmup = wait_before_warmup

    def __call__(self, step):
        # Linear annealing
        if self.type == "linear":
            if step < self.wait_before_warmup:
                return self.annealing_value_start
            elif step < self.end_of_warmup:
                return (step - self.wait_before_warmup) / (
                    self.end_of_warmup - self.wait_before_warmup
                )
            else:
                return self.annealing_value_end

        else:
            # Constant
            return self.annealing_value_end
     
    

if __name__=='__main__':    
    
    main(config=False)
