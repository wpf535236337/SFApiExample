import os
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from scene_classification.utils import scan_all_files


class TestDataset(Dataset):
    def __init__(self, imgdir, transform):

        self.imgdir = imgdir
        self.image_list = scan_all_files(self.imgdir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        input_path = self.image_list[idx].rstrip()
        img = Image.open(input_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(input_path)


if __name__ == '__main__':
    crop_size = (320, 256)
    test_tfms = transforms.Compose([
        transforms.Resize(crop_size),
        # transforms.Resize(int(crop_size / 0.875)),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imgdir = '/Users/01384153/sf_wpf/dataset/test_dataset/zhengjianzhao'
    dataset = TestDataset(imgdir=imgdir, transform=test_tfms)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=96, shuffle=False, num_workers=24)

    for data in tqdm(testloader):
        # get the inputs and assign them to cuda
        inputs = data
        print(inputs)
        # # inputs = inputs.to(device).half() # uncomment for half precision model
        # inputs = inputs.cuda()
        # labels = labels.cuda()

        #

        # outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
