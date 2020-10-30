import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser('Image classification.')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=torch.cuda.is_available(),
                        help='switch cpu/gpu, default:cpu.')
    parser.add_argument('--models_dir',
                        type=str,
                        default='weight/train_log_20200825_060155_snapshot_0020_acc_1.00.pth',
                        help='path of models.')
    parser.add_argument('--img_dir',
                        type=str,
                        default='test_input',
                        help='root of input images')
    parser.add_argument('--out_dir',
                        type=str,
                        default='test_output',
                        help='root of save images')
    parser.add_argument('--class_index',
                        type=str,
                        default='imagenet_class_index.json',
                        help='class dictionary')
    parser.add_argument('--rz_size',
                        type=tuple,
                        default=(320, 256), # (224, 224)
                        help='network input size')
    args = parser.parse_args()

    return args