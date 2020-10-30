import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


def load_model(model, weight_path):

    checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
    pretrained_dict_ = {}

    for k, v in checkpoint.items():
        if 'module' in k:
            pretrained_dict_[k[7:]] = v
        else:
            pretrained_dict_[k] = v

    model.load_state_dict(pretrained_dict_)
    return model


def save_pkl(db_path, db_object):
    with open(db_path, 'wb') as f:
        pickle.dump(db_object, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(db_path):
    with open(db_path, 'rb') as f:
        return pickle.load(f)


def multiprocess(func, lines, worksers_num=10):
    '''
    process multiple images with multithreading
    :param func:
    :param lines:
    :return:
    '''

    res_data = []
    with ThreadPoolExecutor(max_workers=worksers_num) as exe_pool:
        tasks = [
            exe_pool.submit(func, line)
            for line in tqdm(lines)
        ]
    for task in as_completed(tasks):
        res_data.append(task.result())

    return res_data


def scan_all_files(video_root, video_or_img='img'):
    '''
        # scan  multilevel directory
    :param video_root:
    :param video_or_img:  pre-set using [str]: 'video' or 'img'
                         also can set using a [list]: such as [.json]
    :return:
    '''

    if (not os.path.exists(video_root) or not os.path.isdir(video_root)):
        raise Exception("Image Directory [%s] invalid" % video_root)

    def is_video_or_img_file(filename, file_type='video'):

        if file_type == 'video':
            FILE_EXTENSIONS = ['.mp4', '.MP4']

        elif file_type == 'img':
            FILE_EXTENSIONS = [
                '.jpg', '.JPG', '.jpeg',
                '.JPEG', '.png', '.PNG'
            ]
        elif isinstance(file_type, list):
            FILE_EXTENSIONS = file_type

        return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

    file_list = []

    for root, sub_dirs, files in os.walk(video_root):
        for f_n in files:
            video_path = os.path.join(root, f_n)
            if is_video_or_img_file(video_path, video_or_img):
                file_list.append(video_path)

    return file_list


if __name__ == '__main__':
    img_path = 'test_input'
    print(len(scan_all_files(img_path)))