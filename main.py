import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader
import pdb


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    runner = Runner(cfg)
    #pdb.set_trace()
    # cfg = Config(path: configs/clrnet/clr_dla34_culane.py):
    # {
    # 'net': {'type': 'Detector'},
    # 'backbone': {'type': 'DLAWrapper', 'dla': 'dla34', 'pretrained': True},
    # 'num_points': 72,
    # 'max_lanes': 4,
    # 'sample_y': range(589, 230, -20),
    # 'heads': {'type': 'CLRHead', 'num_priors': 192, 'refine_layers': 3, 'fc_hidden_dim': 64, 'sample_points': 36},
    # 'iou_loss_weight': 2.0,
    # 'cls_loss_weight': 2.0,
    # 'xyt_loss_weight': 0.2,
    # 'seg_loss_weight': 1.0,
    # 'work_dirs': 'work_dirs/clr/dla34_culane',
    # 'neck': {'type': 'FPN', 'in_channels': [128, 256, 512], 'out_channels': 64, 'num_outs': 3, 'attention': False},
    # 'test_parameters': {'conf_threshold': 0.4, 'nms_thres': 50, 'nms_topk': 4},
    # 'epochs': 15,
    # 'batch_size': 24,
    # 'optimizer': {'type': 'AdamW', 'lr': 0.0006},
    # 'total_iter': 55545,
    # 'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 55545},
    # 'eval_ep': 3,
    # 'save_ep': 10,
    # 'img_norm': {'mean': [103.939, 116.779, 123.68], 'std': [1.0, 1.0, 1.0]},
    # 'ori_img_w': 1640,
    # 'ori_img_h': 590,
    # 'img_w': 800,
    # 'img_h': 320,
    # 'cut_height': 270,
    # 'train_process': [{'type': 'GenerateLaneLine', 'transforms': [{'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}, {'name': 'HorizontalFlip', 'parameters': {'p': 1.0}, 'p': 0.5}, {'name': 'ChannelShuffle', 'parameters': {'p': 1.0}, 'p': 0.1}, {'name': 'MultiplyAndAddToBrightness', 'parameters': {'mul': (0.85, 1.15), 'add': (-10, 10)}, 'p': 0.6}, {'name': 'AddToHueAndSaturation', 'parameters': {'value': (-10, 10)}, 'p': 0.7}, {'name': 'OneOf', 'transforms': [{'name': 'MotionBlur', 'parameters': {'k': (3, 5)}}, {'name': 'MedianBlur', 'parameters': {'k': (3, 5)}}], 'p': 0.2}, {'name': 'Affine', 'parameters': {'translate_percent': {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, 'rotate': (-10, 10), 'scale': (0.8, 1.2)}, 'p': 0.7}, {'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}]}, {'type': 'ToTensor', 'keys': ['img', 'lane_line', 'seg']}],
    # 'val_process': [{'type': 'GenerateLaneLine', 'transforms': [{'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}], 'training': False}, {'type': 'ToTensor', 'keys': ['img']}],
    # 'dataset_path': './data/CULane',
    # 'dataset_type': 'CULane',
    # 'dataset': {'train': {'type': 'CULane', 'data_root': './data/CULane', 'split': 'train', 'processes': [{'type': 'GenerateLaneLine', 'transforms': [{'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}, {'name': 'HorizontalFlip', 'parameters': {'p': 1.0}, 'p': 0.5}, {'name': 'ChannelShuffle', 'parameters': {'p': 1.0}, 'p': 0.1}, {'name': 'MultiplyAndAddToBrightness', 'parameters': {'mul': (0.85, 1.15), 'add': (-10, 10)}, 'p': 0.6}, {'name': 'AddToHueAndSaturation', 'parameters': {'value': (-10, 10)}, 'p': 0.7}, {'name': 'OneOf', 'transforms': [{'name': 'MotionBlur', 'parameters': {'k': (3, 5)}}, {'name': 'MedianBlur', 'parameters': {'k': (3, 5)}}], 'p': 0.2}, {'name': 'Affine', 'parameters': {'translate_percent': {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, 'rotate': (-10, 10), 'scale': (0.8, 1.2)}, 'p': 0.7}, {'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}]}, {'type': 'ToTensor', 'keys': ['img', 'lane_line', 'seg']}]},
    # 'val': {'type': 'CULane', 'data_root': './data/CULane', 'split': 'test', 'processes': [{'type': 'GenerateLaneLine', 'transforms': [{'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}], 'training': False}, {'type': 'ToTensor', 'keys': ['img']}]}, 'test': {'type': 'CULane', 'data_root': './data/CULane', 'split': 'test', 'processes': [{'type': 'GenerateLaneLine', 'transforms': [{'name': 'Resize', 'parameters': {'size': {'height': 320, 'width': 800}}, 'p': 1.0}], 'training': False}, {'type': 'ToTensor', 'keys': ['img']}]}}, 'workers': 10, 'log_interval': 500, 'num_classes': 5, 'ignore_label': 255, 'bg_weight': 0.4, 'lr_update_by_epoch': False, 'gpus': 1, 'load_from': 'culane_dla34.pth', 'resume_from': None, 'finetune_from': None, 'view': False, 'seed': 0, 'work_dir': 'work_dirs/clr/dla34_culane/20231127_213252_lr_6e-04_b_24'}

    if args.validate:
        runner.validate()
    elif args.test:
        runner.test()
    else:
        runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
