import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from scene_classification.mobilenet import mobilenet_v2
from scene_classification.scene_options import parse_args
from scene_classification.dataset import TestDataset
from scene_classification.utils import load_model, scan_all_files


class ImageClassification(object):
    def __init__(self, clc_args):
        self.args = clc_args
        # environment
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.use_gpu = self.args.use_gpu

        # model
        # self.model = resnet18(pretrained=False)
        self.model = mobilenet_v2(pretrained=False, num_classes=5)
        self.model = load_model(self.model, weight_path=self.args.models_dir)
        self.model.eval().cuda() if self.use_gpu else self.model.eval()

        # dataloader
        self.rz_size = self.args.rz_size
        self.test_tfms = transforms.Compose([
            transforms.Resize(self.rz_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # class_number
        # self.classes_index = {int(k): v[1] for k, v in
                              # (json.load(open(self.args.class_index))).items()}
        class_list = ['cargo', 'door', 'error', 'guard', 'shelf']
        self.classes_index = dict(zip(range(len(class_list)), class_list))

    def preprocess(self, img):
        # resize netwok input size
        if isinstance(img, str):
            img = cv2.imread(img, 1)
        else:
            img = img.copy()
        img = cv2.resize(img, self.rz_size)
        # turn to RGB
        img = img[:, :, ::-1]
        img = np.array(img, dtype=np.float32) / 255
        # normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img -= mean
        img /= std
        # turn to NxCxHxW tensor
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        img_tensor = img_tensor.cuda() if self.use_gpu else img_tensor
        return img_tensor

    def postprocess(self, img_src, net_output, image_path):

        # visualize the result
        prob, class_num = F.softmax(net_output, dim=1).max(1)
        clc_label = self.classes_index[class_num.item()]

        # save_root = os.path.join(args.out_dir, clc_label)
        # os.makedirs(save_root, exist_ok=True)
        # shutil.move(image_path, save_root)

        # cv2.putText(img_src, "Label: {}".format(clc_label), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
        # cv2.putText(img_src, "Prob: {:.2f}%".format(prob.item() * 100), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
        #             (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(self.args.out_dir, os.path.basename(image_path)), img_src)
        # cv2.imshow("Classification", img_src)
        # cv2.waitKey()

        return clc_label

    @torch.no_grad()
    def predict(self, image_path):
        if isinstance(image_path, str):
            img_src = cv2.imread(image_path, 1)
        else:
            img_src = image_path.copy()
        image_input = self.preprocess(img_src)
        net_output = self.model(image_input)
        prob, class_num = F.softmax(net_output, dim=1).max(1)
        clc_label = self.classes_index[class_num.item()]
        # clc_label = self.postprocess(img_src, net_output, image_path)
        return prob.item(), clc_label

    @torch.no_grad()
    def batch_predict(self, img_dir):
        dataset = TestDataset(imgdir=img_dir, transform=self.test_tfms)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=24)
        img_names = []
        class_number = []

        len_testloader = len(testloader)
        for i, data in enumerate(testloader):
            inputs, names = data
            inputs = inputs if not self.use_gpu else inputs.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            class_number.extend(predicted.cpu().numpy())
            img_names.extend(names)
            test_info = '==============================Progress[{}/{}]'.format(i+1, len_testloader)
            print(test_info)
        # ==================================save result======================================================
        dataframe = pd.DataFrame({'name': img_names, 'class': class_number})
        test_csv_path = os.path.join(self.args.out_dir, os.path.splitext(os.path.basename(args.models_dir))[0] + '.csv')
        dataframe.to_csv(test_csv_path, index=False, sep=',')
        print('result has been saved {}'.format(test_csv_path))


if __name__ == "__main__":
    # select the gpu number
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # establish the predictor
    args = parse_args()
    cls = ImageClassification(args)
    # start to infer
    for img_path in scan_all_files(args.img_dir):
        prob, label = cls.predict(img_path)
        print('class name:', label, ',probability', prob)
    # img_lines = scan_all_files(args.img_dir)
    # multiprocess(cls.predict, img_lines)
    # batch inference
    # cls.batch_predict(args.img_dir)
