import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def listdirInMac(path: str) -> list:
    files = os.listdir(path)
    for file in files:
        if file.startswith('.') and os.path.isfile(os.path.join(path, file)):
            files.remove(file)
    return files

class CDDataset(Dataset):
    def __init__(self, dir: str) -> None:
        self.imgA_path = dir + '/A/'
        self.imgB_path = dir + '/B/'
        self.label_path = dir + '/label/'
        self.file_names = listdirInMac(self.imgA_path)

    def __getitem__(self, item):
        file_name = self.file_names[item]
        imgA = cv2.imread(self.imgA_path + file_name)
        imgB = cv2.imread(self.imgB_path + file_name)
        label = cv2.imread(self.label_path + file_name)

        label = label // 255
        label = label[:,:,0]

        [imgA, imgB] = [TF.to_tensor(img) for img in [imgA, imgB]]
        [imgA, imgB] = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                for img in [imgA, imgB]]
        label = torch.from_numpy(np.array(label, np.uint8))

        return imgA, imgB, label

    def __len__(self):

        return len(self.file_names)

