import pprint
import argparse
import tqdm
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import json
import Datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.model import DroneGuard as get_model
import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --cfg experiments/sha/sha_wresnet.yaml --model-file output/shanghai/sha_wresnet/shanghai.pth GPUS [3]
# --cfg experiments/ped2/ped2_wresnet.yaml --model-file output/ped2/ped2_wresnet/ped2.pth GPUS [3]
def parse_args():
    parser = argparse.ArgumentParser(description='Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='pretrained/shanghaitech.pth', type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False       # TODO ? False
    config.freeze()

    # --- START OF DEVICE FIX ---
    
    # 1. Define the device based on availability (MPS for M-series Mac, otherwise CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Running on device: {device}')
    
    gpus = [(config.GPUS[0])]
    model = get_model(config)
    logger.info('Model: {}'.format(model.get_name()))
    
    # 2. Move model to the selected device, replacing nn.DataParallel().cuda()
    # DataParallel is ignored for single-device (MPS/CPU) use.
    model = model.to(device)
    
    # --- END OF DEVICE FIX ---
    
    logger.info('Epoch: '.format(args.model_file))

    
    # load model
    # Load model state dictionary, mapping it to the correct device
    state_dict = torch.load(args.model_file, map_location=device) 
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    
    # Handle both DataParallel (has .module) and non-DataParallel models
    target_model = model.module if hasattr(model, 'module') else model
    target_model.load_state_dict(state_dict)

    test_dataset = eval('Datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    if config.DATASET.TYPE == "ground":
        mat=anomaly_util.get_labels(config.DATASET.DATASET, 'Drone-Guard/Datasets', 'Drone-Guard/data')
    else :
        mat=anomaly_util.get_drone_labels(config.DATASET.DATASET, 'Drone-Guard/Datasets')
    
    # Pass the device to inference function
    psnr_list = inference(config, test_loader, model, args) 
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    # --- Prepare flattened scores and labels using the same normalization as calculate_auc ---
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    scores = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.int64)

    for i in range(len(psnr_list)):
        arr = np.array(psnr_list[i], dtype=np.float64)
        # use the same normalization as in anomaly_util.calculate_auc
        score = anomaly_util.anomaly_score(arr, np.max(arr), np.min(arr))
        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)

    # Save flattened arrays to the experiment output folder so they can be inspected/used
    y_score_path = os.path.join(final_output_dir, 'y_score.npy')
    y_true_path = os.path.join(final_output_dir, 'y_true.npy')
    np.save(y_score_path, scores)
    np.save(y_true_path, labels)
    logger.info(f'Saved scores to: {y_score_path}')
    logger.info(f'Saved labels to: {y_true_path}')

    # Calculate final AUC (re-using existing utility)
    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)
    logger.info(f'AUC: {auc * 100:.1f}% ')

    # Generate and save plots + metrics JSON
    metrics_data = save_metrics_and_plots(final_output_dir, scores, labels, logger)
    logger.info(f'Metrics and plots saved to {final_output_dir}')



def inference(config, data_loader, model, args, quiet=False):
    loss_func_mse = nn.MSELoss(reduction='none')
    
    model.eval()
    psnr_list = []
    
    # --- START OF INFERENCE DEVICE FIX ---
    # Get the device the model is currently running on
    device = next(model.parameters()).device
    # --- END OF INFERENCE DEVICE FIX ---
    
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if(not(quiet)):
              print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []
            # compute the output
            video, video_name = train_util.decode_input(input=data, train=False)
            
            # Move data to the same device as the model
            video = [frame.to(device) for frame in video] 
            
            for f in tqdm.tqdm(range(len(video) - fp), disable=quiet):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
           
    return psnr_list 


def save_metrics_and_plots(final_output_dir, scores, labels, logger):
    """
    Generate ROC/PR curves and save metrics JSON.
    Supports both project convention (pos_label=0) and conventional (pos_label=1).
    """
    try:
        from sklearn import metrics
    except ImportError:
        logger.warning('sklearn not available; skipping metrics plots')
        return

    # Project convention: pos_label=0
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auc_reported = float(metrics.auc(fpr, tpr))
    precision_r, recall_r, _ = metrics.precision_recall_curve(labels, scores, pos_label=0)
    ap_reported = float(metrics.average_precision_score(labels, scores, pos_label=0))

    # Conventional: pos_label=1 with inverted score
    scores_inv = 1.0 - scores
    fpr_c, tpr_c, _ = metrics.roc_curve(labels, scores_inv, pos_label=1)
    auc_conv = float(metrics.auc(fpr_c, tpr_c))
    precision_c, recall_c, _ = metrics.precision_recall_curve(labels, scores_inv, pos_label=1)
    ap_conv = float(metrics.average_precision_score(labels, scores_inv, pos_label=1))

    # Find best threshold by F1
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.unique(scores_inv):
        y_pred = (scores_inv >= thr).astype(int)
        prec = metrics.precision_score(labels, y_pred, zero_division=0)
        rec = metrics.recall_score(labels, y_pred, zero_division=0)
        f1 = metrics.f1_score(labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    y_pred_best = (scores_inv >= best_thr).astype(int)
    cm = metrics.confusion_matrix(labels, y_pred_best)
    TN, FP, FN, TP = cm.ravel()
    acc = metrics.accuracy_score(labels, y_pred_best)

    # Save ROC curves
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC={auc_reported:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (project convention, pos_label=0)')
    plt.legend()
    roc_path = os.path.join(final_output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved ROC curve to {roc_path}')

    # Save PR curves
    plt.figure(figsize=(6, 6))
    plt.plot(recall_r, precision_r, label=f'AP={ap_reported:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (project convention, pos_label=0)')
    plt.legend()
    pr_path = os.path.join(final_output_dir, 'pr_curve.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved PR curve to {pr_path}')

    # Save summary bar chart
    metric_names = ['AUC\n(reported)', 'AP\n(reported)', 'AUC\n(conventional)', 'AP\n(conventional)', 'Best F1', 'Accuracy']
    metric_values = [auc_reported, ap_reported, auc_conv, ap_conv, best_f1, acc]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1)
    for bar, v in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    plt.title('Metrics Summary')
    plt.ylabel('Score')
    summary_path = os.path.join(final_output_dir, 'metrics_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved metrics summary to {summary_path}')

    # Save JSON metrics
    metrics_data = {
        'reported_convention_pos_label_0': {
            'AUC': auc_reported,
            'Average_Precision': ap_reported
        },
        'conventional_pos_label_1_inverted_score': {
            'AUC': auc_conv,
            'Average_Precision': ap_conv,
            'Best_threshold_by_F1': best_thr,
            'Best_F1': best_f1,
            'TP': int(TP),
            'FP': int(FP),
            'TN': int(TN),
            'FN': int(FN),
            'Accuracy': float(acc)
        }
    }
    metrics_json_path = os.path.join(final_output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f'Saved metrics JSON to {metrics_json_path}')

    return metrics_data


if __name__ == '__main__':
    main()