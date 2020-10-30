import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser('SFApiExample.')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=torch.cuda.is_available(),
                        help='switch cpu/gpu, default:cpu.')
    parser.add_argument('--models_dir',
                        type=str,
                        default='scene_classification/weight/train_log_20200825_060155_snapshot_0020_acc_1.00.pth',
                        help='path of models.')
    parser.add_argument('--img_dir',
                        type=str,
                        default='scene_classification/test_input',
                        help='root of input images')
    parser.add_argument('--out_dir',
                        type=str,
                        default='scene_classification/test_output',
                        help='root of save images')
    parser.add_argument('--hard_example_dir',
                        type=str,
                        default='hard_example',
                        help='root of save images')
    parser.add_argument('--rz_size',
                        type=tuple,
                        default=(320, 256), # (224, 224)
                        help='network input size')
    args = parser.parse_args()

    return args