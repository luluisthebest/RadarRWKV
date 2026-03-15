import torch
import numpy as np
# from .metrics import GetFullMetrics, Metrics, RA_to_cartesian_box
import pkbar
import cv2
import os
import polarTransform
from .distributed_utils import reduce_value
# from utils.util import process_predictions_FFT, worldToImage
from dataset.data_preparation import radarcube_lidarcube_loss, lidarpc_to_lidarcube, lidarpc_to_lidarcube_multiclass, cube_to_pointcloud, cube_to_pointcloud_multiclass
from .compute_metrics import compute_pd_pfa_gpu, compute_pd_pfa, compute_pd_pfa_gpu_multiclass
from utils.compute_metrics import compute_chamfer_distance_gpu, hoyer_metric, compute_chamfer_distance


range_min = 5
range_max = 100
def run_evaluation(name,epoch,net,loader,encoder,check_perf=False, T=1, loss_ce=None, loss_fl=None, detection_loss=None,
                   segmentation_loss=None,losses_params=None, doppler=False, adaptive_loss=None, multiclass=False, config=None,
                   lambda_klg=0.0, lambda_kld=0.0, kl_factor=0.0, weights=None):


    net.eval()
    running_loss = 0.0
    pd_total = 0.0
    pfa_total = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda')   #.float()
        label_map = data[1].to('cuda').float()

        with torch.set_grad_enabled(False):
            # outputs, z_out, latent_loss = net(inputs)
            outputs, kl_g_list, kl_d_list, alpha_list = net(inputs, kl_factor=kl_factor)
        
        alpha_list_detached = [alpha.detach() for alpha in alpha_list]
        classif_loss = radarcube_lidarcube_loss(outputs, label_map, config, multiclass, weights)
        klg = kl_factor * (weighted_mean(kl_g_list, True) * lambda_klg)
        kld = kl_factor * (weighted_mean(kl_d_list, True) * lambda_kld)

        loss = classif_loss + klg + kld      
            
        loss_reduced = reduce_value(loss)
        # statistics
        running_loss += loss_reduced.item()   * inputs.size(0)

        if(check_perf):
            # radar_cube_out = outputs.sigmoid().squeeze().cpu().detach().numpy()
            outputs = torch.squeeze(outputs)
            
            pd, pfa = compute_pd_pfa_gpu(label_map, outputs)
            pd_total += pd.item() * inputs.size(0)
            pfa_total += pfa.item() * inputs.size(0)

        kbar.update(i)        
        # break  # for debug

    return {'loss':running_loss/len(loader.dataset), 
            'loss_cls':classif_loss, 
            'klg':klg, 
            'kld':kld, 
            'Pd':pd_total/len(loader.dataset), 
            'Pfa':pfa_total/len(loader.dataset)}


def run_FullEvaluation(name,net,loader,check_perf=False, T=1, loss_ce=None, loss_fl=None, detection_loss=None,
                   segmentation_loss=None,losses_params=None, doppler=False, adaptive_loss=None, multiclass=False, config=None,
                   lambda_klg=0.0, lambda_kld=0.0, kl_factor=0.0):

    net.eval()
    radar_distance = 0.0
    prc_total = 0.0
    pfa_total = 0.0
    
    count = 0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda')   #.float()
        
        with torch.set_grad_enabled(False):
            # print(inputs.dtype)
            outputs, kl_g_list, kl_d_list, alpha = net(inputs, kl_factor=kl_factor)
        
        lidar_pc = np.load(data[2]['gt_path'][0])
        lidar_pc[:, 1] = -lidar_pc[:, 1]
        lidar_cube = lidarpc_to_lidarcube(lidar_pc, config)
        radar_pc = cube_to_pointcloud(outputs[:,0,...].cpu().squeeze(), config, inputs.cpu().squeeze(),'radar')
        radar_pc[:, 2] = -radar_pc[:, 2]
        if radar_pc.shape[1] == 4:
            radar_pc = radar_pc[:, :-1] 

        # radar_distance = radar_distance + compute_chamfer_distance_gpu(lidar_pc, radar_pc)
        radar_distance = radar_distance + compute_chamfer_distance(lidar_pc, radar_pc)
        radar_cube_sized = lidarpc_to_lidarcube(radar_pc, config)

        pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube_sized)
        prc_total = prc_total + pd_radar_aux
        pfa_total = pfa_total + pfa_radar_aux

        count = count + 1
        kbar.update(i)
    
    print('Pd CFAR: ' + str(prc_total / count))
    print('Pfa CFAR: ' + str(pfa_total / count))
    print('Distance CFAR: ' + str(radar_distance / count))



def weighted_mean(kl_list, weighted_mean=False):
    if weighted_mean:
        weights = [i for i in range(1, len(kl_list) + 1)]
        # weights = [(2**i) for i in range(0, len(kl_list))]
    else:  # Equal weighted Mean
        weights = [1 for i in range(0, len(kl_list))]

    weights = [weight / (sum(weights)) for weight in weights]
    return sum([torch.mean(kl_layer) * weights[i] for i, kl_layer in enumerate(kl_list)])
