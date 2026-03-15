import os
import json
import argparse
import torch
import random
import numpy as np
from dataset.dataset import RADIal
from model.MVIBRadarRWKV_dp import SparseRadarRWKV
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
# from utils.distributed_eval import run_FullEvaluation
from utils.eval_vib import run_FullEvaluation
import cv2
from torch.utils.tensorboard import SummaryWriter
from dataset.rad_cube_loader import RADCUBE_DATASET
from torch.utils.data import DataLoader

def main(config, checkpoint_filename,difficult, num_frames):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dop = False
    dataset = 'radelft'
    if dataset == 'radelft':
        multiclass = True
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

        test_dataset = RADCUBE_DATASET(mode='test', params=config, multiclss=multiclass)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                shuffle=False,
                                batch_size=config['test']['batch_size'], 
                                num_workers=config['test']['num_workers'],
                                pin_memory=True)
    else:
        enc = ra_encoder(geometry = config['dataset']['geometry'], 
                            statistics = config['dataset']['statistics'],
                            regression_layer = 3 if dop else 2)
        
        dataset = RADIal(root_dir = config['dataset']['root_dir'],
                            statistics= config['dataset']['statistics'],
                            encoder=enc.encode,
                            difficult=difficult,
                            perform_FFT=config['data_mode'])

        train_loader, train_sampler, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    # Create the model
    if config['name'] == 'radarrwkv':
        # net = RadarRWKV2()
        net = SparseRadarRWKV(device=device, multiclass=multiclass)

    net.to(device)
    net = torch.nn.DataParallel(net)

    print('===========  Loading the model ==================:')
    # dict = torch.load(checkpoint_filename, weights_only=False)
    # net.load_state_dict(dict['net_state_dict'])
    
    writer = SummaryWriter(os.path.dirname(checkpoint_filename))
    # cal_grad(net, writer)
    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(name=config['name'], net=net, multiclass=True, loader=test_loader, config=config)
    # os.makedirs('',)

       
       
if __name__=='__main__':
    # PARSE THE ARGS
    
    parser = argparse.ArgumentParser(description='4D radar cube Training')
    parser.add_argument('-c', '--config', default='./configs/RaDelft.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=\
                        './radarrwkv2_epoch99_val_loss_332.2781_AP_0.9898_AR_0.9238_IOU_0.6748.pth', \
                        type=str, help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', default=True, action='store_true')
    parser.add_argument('--num_frames', default=5)
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult, args.num_frames)
