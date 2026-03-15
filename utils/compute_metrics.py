import numpy as np
from dataset.data_preparation import *
from dataset.rad_cube_loader import RADCUBE_DATASET_TIME, RADCUBE_DATASET
from sklearn.neighbors import KDTree
import os
from typing import Optional
import torch.nn as nn


def compute_metrics(params):
    """
    Compute the Chamfer distance and the Pfa and Pd on all the frames.
    This is used for the single frame version.

    :param params: default parameters from data_preparation.py
    :return: Nothing, only prints the result
    """
    # Create Loader
    val_dataset = RADCUBE_DATASET(mode='test', params=params)

    cfar_distance = 0.0
    radar_distance = 0.0
    count = 0.0
    pd_cfar = 0.0
    pd_radar = 0.0
    pfa_cfar = 0.0
    pfa_radar = 0.0

    for dataset_dict in val_dataset.data_dict.values():

        # Generate lidar_cube
        lidar = dataset_dict['gt_path']
        lidarpc = np.load(lidar)
        lidar_cube = lidarpc_to_lidarcube(lidarpc, params)

        # Load radar point clouds
        cfar = dataset_dict['cfar_path']
        network_output = cfar.replace('radar_ososos', 'network')
        radarpc = np.load(network_output)
        if radarpc.shape[1] == 4:
            radarpc = radarpc[:, :-1]  # Remove speed to compute metrics
        radarpc[:, 1] = -radarpc[:, 1]

        # Compute Metrics
        radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)
        radar_cube = lidarpc_to_lidarcube(radarpc, params)
        pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)
        pd_radar = pd_radar + pd_radar_aux
        pfa_radar = pfa_radar + pfa_radar_aux

        '''
        Uncomment if the CFAR metrics wants to be calculated
        
        cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
        cfarpc = cfarpc[:,0:3]
        cfar_distance = cfar_distance + data_preparation.compute_chamfer_distance(lidarpc,cfarpc)
        cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)
        pd_cfar_aux, pfa_cfar_aux = data_preparation.compute_pd_pfa(lidar_cube, cfar_cube)
        pd_cfar = pd_cfar + pd_cfar_aux
        pfa_cfar = pfa_cfar + pfa_cfar_aux
        '''

        count = count + 1

        if count % 10 == 0:
            print(str(count))

    print('Pd CFAR: ' + str(pd_cfar / count))
    print('Pd NET: ' + str(pd_radar / count))
    print('----------')
    print('Pfa CFAR: ' + str(pfa_cfar / count))
    print('Pfa Net: ' + str(pfa_radar / count))
    print('----------')
    print('Distance CFAR: ' + str(cfar_distance / count))
    print('Distance Net: ' + str(radar_distance / count))


def compute_metrics_time(params):
    """
    Compute the Chamfer distance and the Pfa and Pd on all the frames.
    This is used for the multi frame version.

    :param params: default parameters from data_preparation.py
    :return: Nothing, only prints the result
    """
    # Create Loader
    val_dataset = RADCUBE_DATASET_TIME(mode='test',  params=params)

    cfar_distance = 0
    radar_distance = 0
    count = 0
    pd_cfar = 0
    pd_radar = 0
    pfa_cfar = 0
    pfa_radar = 0

    for dataset_dict in val_dataset.data_dict.values():
        for t in dataset_dict.keys():

            # Generate lidar_cube
            lidar = dataset_dict[t]['gt_path']
            lidarpc = np.load(lidar)
            lidarpc[:, 1] = -lidarpc[:, 1]

            lidar_cube = lidarpc_to_lidarcube(lidarpc, params)

            # Load radar point clouds
            cfar = dataset_dict[t]['cfar_path']
            network_output = cfar.replace('radar_ososos', 'network')
            if not os.path.isfile(network_output):
                continue
            radarpc = np.load(network_output)
            if radarpc.shape[1] == 4:
                radarpc = radarpc[:, :-1]           #Remove speed to compute metrics

            # Compute Metrics
            radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)
            radar_cube = lidarpc_to_lidarcube(radarpc, params)
            pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)
            pd_radar = pd_radar + pd_radar_aux
            pfa_radar = pfa_radar + pfa_radar_aux

            '''
            Uncomment if the CFAR metrics wants to be calculated
            
            cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
            cfarpc = cfarpc[:,0:3]
            cfar_distance = cfar_distance + compute_chamfer_distance(lidarpc,cfarpc)
            cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)
            pd_cfar_aux, pfa_cfar_aux = compute_pd_pfa(lidar_cube, cfar_cube)
            pd_cfar = pd_cfar + pd_cfar_aux
            pfa_cfar = pfa_cfar + pfa_cfar_aux
            '''
            count = count + 1

            if count % 10 == 0:
                print(str(count))

    print('Pd CFAR: ' + str(pd_cfar / count))
    print('Pd NET: ' + str(pd_radar / count))
    print('----------')
    print('Pfa CFAR: ' + str(pfa_cfar / count))
    print('Pfa Net: ' + str(pfa_radar / count))
    print('----------')
    print('Distance CFAR: ' + str(cfar_distance / count))
    print('Distance Net: ' + str(radar_distance / count))


def compute_chamfer_distance(point_cloud1, point_cloud2):
    """
    Compute the Chamfer distance between two set of points

    :param point_cloud1: the first set of points
    :param point_cloud2: the second set of points
    :return: the Chamfer distance
    """
    tree1 = KDTree(point_cloud1, metric='euclidean')
    tree2 = KDTree(point_cloud2, metric='euclidean')
    distances1, _ = tree1.query(point_cloud2)
    distances2, _ = tree2.query(point_cloud1)
    av_dist1 = np.sum(distances1) / np.size(distances1)
    av_dist2 = np.sum(distances2) / np.size(distances2)
    dist = av_dist1 + av_dist2

    return dist

def compute_chamfer_distance_gpu(point_cloud1, point_cloud2):
    """
    Compute the Chamfer distance between two set of points

    :param point_cloud1: the first set of points
    :param point_cloud2: the second set of points
    :return: the Chamfer distance
    """
    point_cloud1 = torch.from_numpy(point_cloud1).float().cuda()
    point_cloud2 = torch.from_numpy(point_cloud2).float().cuda()
    # tree1 = KDTree(point_cloud1, metric='euclidean')
    # tree2 = KDTree(point_cloud2, metric='euclidean')
    # distances1, _ = tree1.query(point_cloud2)
    # distances2, _ = tree2.query(point_cloud1)     # CPU
    dist_matrix = torch.cdist(point_cloud1, point_cloud2)  # O(N²) 但并行化
    dist1 = dist_matrix.min(dim=1)[0]                      # 每行的最小值
    dist2 = dist_matrix.min(dim=0)[0]                      # 每列的最小值
    # av_dist1 = np.sum(distances1) / np.size(distances1)
    # av_dist2 = np.sum(distances2) / np.size(distances2)
    av_dist1 = torch.mean(dist1)
    av_dist2 = torch.mean(dist2)
    dist = av_dist1 + av_dist2

    return dist

def compute_pd_pfa(ground_truth, prediction):
    """
    Compute the Pd and Pfa between two 3D cubes

    :param ground_truth: the 3D cube to compare. Usually the lidar cube.
    :param prediction: the estimated 3D cube. Usually the rada cube.
    :return: the Pd and Pfa
    """
    # Flatten the matrices to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = prediction.flatten()

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((ground_truth_flat == 1) & (prediction_flat == 1))
    FP = np.sum((ground_truth_flat == 0) & (prediction_flat == 1))
    FN = np.sum((ground_truth_flat == 1) & (prediction_flat == 0))

    # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + (ground_truth_flat.size - TP - FN)) if (FP + (ground_truth_flat.size - TP - FN)) > 0 else 0

    return TPR, FPR

def compute_pd_pfa_gpu(ground_truth, prediction):
    """
    Compute the Pd and Pfa between two 3D cubes

    :param ground_truth: the 3D cube to compare. Usually the lidar cube.
    :param prediction: the estimated 3D cube. Usually the rada cube.
    :return: the Pd and Pfa
    """
    # ground_truth = torch.from_numpy(ground_truth).float().cuda()
    # prediction = torch.from_numpy(prediction).float().cuda()
    prediction = prediction > 0.5
    prediction = prediction[..., :-12, 8:-8]
    # Flatten the matrices to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = prediction.flatten()

    total_samples = torch.tensor(ground_truth_flat.numel(), dtype=torch.float32).cuda()

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = torch.sum((ground_truth_flat == 1) & (prediction_flat == 1))
    FP = torch.sum((ground_truth_flat == 0) & (prediction_flat == 1))
    FN = torch.sum((ground_truth_flat == 1) & (prediction_flat == 0))

    # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    # FPR = FP / (FP + (ground_truth_flat.size - TP - FN)) if (FP + (ground_truth_flat.size - TP - FN)) > 0 else 0
    FPR = FP / (FP + (total_samples - TP - FN)) if (FP + (total_samples - TP - FN)) > 0 else 0

    return TPR, FPR

def compute_iou_dice(ground_truth, prediction):

    hist = _fast_hist(ground_truth.flatten().astype(np.int64), prediction.flatten().astype(np.int64), 2)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)

    return avg_jacc, avg_dice


def compute_pd_pfa_gpu_multiclass(ground_truth, prediction):
    """
    Compute the Pd and Pfa between two 3D cubes

    :param ground_truth: the 3D cube to compare. Usually the lidar cube.
    :param prediction: the estimated 3D cube. Usually the rada cube.
    :return: the Pd and Pfa
    """
    # ground_truth = torch.from_numpy(ground_truth).float().cuda()
    # prediction = torch.from_numpy(prediction).float().cuda()
    probs = torch.softmax(prediction, dim=1)  # 形状: (B, 5, 34, H, W)
    pred_classes = torch.argmax(probs, dim=1)  # 形状: (B, 34, H, W)
    pred_probs, _ = torch.max(probs, dim=1)  # 形状: (B, 34, H, W)
    pred_classes = pred_classes[..., :-12, 8:-8]
    # Flatten the matrices to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = pred_classes.flatten()

    total_samples = torch.tensor(ground_truth_flat.numel(), dtype=torch.float32).cuda()

    num_class = 5

    # # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    # TP = torch.sum((ground_truth_flat == 1) & (prediction_flat == 1))
    # FP = torch.sum((ground_truth_flat == 0) & (prediction_flat == 1))
    # FN = torch.sum((ground_truth_flat == 1) & (prediction_flat == 0))

    # # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    # TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    # # FPR = FP / (FP + (ground_truth_flat.size - TP - FN)) if (FP + (ground_truth_flat.size - TP - FN)) > 0 else 0
    # FPR = FP / (FP + (total_samples - TP - FN)) if (FP + (total_samples - TP - FN)) > 0 else 0
    metrics = {
        'per_class': {},
        'average': {},
        'overall': {}
    }
    
    # 计算每个类别的指标
    for class_id in range(num_class):
        # 二值化：当前类别为正类，其他为负类
        gt_binary = (ground_truth_flat == class_id)
        pred_binary = (prediction_flat == class_id)
        
        # 计算TP, FP, FN, TN
        TP = torch.sum(gt_binary & pred_binary).float()
        FP = torch.sum(~gt_binary & pred_binary).float()
        FN = torch.sum(gt_binary & ~pred_binary).float()
        TN = torch.sum(~gt_binary & ~pred_binary).float()
        
        # total_temp = TP+FP+FN+TN   #看是否等于总点数？？？？是
        # 计算各项指标
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 查全率/召回率
        FPR = FP / (FP + (total_samples - TP - FN)) if (FP + (total_samples - TP - FN)) > 0 else 0.0  # 假正率
        
        # 准确率
        accuracy = (TP + TN) / total_samples
        
        # 精确率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
        # F1分数
        f1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0.0
        
        # IoU（交并比）
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        
        # 存储该类别指标
        metrics['per_class'][class_id] = {
            'class_name': get_class_name(class_id),
            'TP': float(TP),
            'FP': float(FP),
            'FN': float(FN),
            'TN': float(TN),
            'TPR': float(TPR),
            'FPR': float(FPR),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'f1_score': float(f1),
            'iou': float(iou)
        }
    
    # 计算平均指标
    avg_tpr = np.mean([metrics['per_class'][c]['TPR'] for c in range(0, num_class)])  # 排除背景类
    avg_fpr = np.mean([metrics['per_class'][c]['FPR'] for c in range(0, num_class)])
    avg_accuracy = np.mean([metrics['per_class'][c]['accuracy'] for c in range(0, num_class)])
    avg_precision = np.mean([metrics['per_class'][c]['precision'] for c in range(0, num_class)])
    avg_f1 = np.mean([metrics['per_class'][c]['f1_score'] for c in range(0, num_class)])
    avg_iou = np.mean([metrics['per_class'][c]['iou'] for c in range(0, num_class)])  # mIoU
    
    # 整体准确率

    overall_acc = torch.sum(ground_truth_flat == prediction_flat).float() / total_samples
    
    metrics['overall'] = {
        'overall_accuracy': overall_acc
    }
    metrics['average'] = {
        'mean_TPR': avg_tpr,
        'mean_FPR': avg_fpr,
        'mean_accuracy': avg_accuracy,
        'mean_precision': avg_precision,
        'mean_f1_score': avg_f1,
        'mean_iou': avg_iou,
        'overall_accuracy': float(overall_acc)
    }
    
    return metrics

def get_class_name(class_id):
    """根据类别ID获取类别名称"""
    class_names = {
        0: 'background noise',
        1: 'building',
        2: 'pedestrian',
        3: 'car',
        4: 'bicycle'
    }
    return class_names.get(class_id, f'class_{class_id}')


def hoyer_metric(z):
    b, K, h, w = z.shape
    K = torch.tensor(K)
    
    l1_norm = torch.norm(z, p=1, dim=1, keepdim=True)  # [B, 1, H, W]
    l2_norm = torch.norm(z, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

    sparsity_score = (torch.sqrt(K) - l1_norm / l2_norm) / (torch.sqrt(K) - 1)
    hoyer_metric_value = torch.mean(sparsity_score)

    return hoyer_metric_value



EPS = 1e-10
def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return np.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(float)
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = np.diag(hist)
    A = hist.sum(axis=1)
    B = hist.sum(axis=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.

    Args:
        hist: confusion matrix.

    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = np.diag(hist)
    A = hist.sum(axis=1)
    B = hist.sum(axis=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.

    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten().astype(np.int64), p.flatten().astype(np.int64), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice
