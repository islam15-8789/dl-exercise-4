from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode, transform=None):
        self.data = data
        self.mode = mode
        self._transform = transform    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = str(Path("./", self.data.iloc[index, 0]))
        image_label = torch.as_tensor(
            [self.data.iloc[index, 1], self.data.iloc[index, 2]]
        )
        image = imread(image_path)
        image = gray2rgb(image=image)
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=torch.tensor(train_mean),
                    std=torch.tensor(train_std),
                    inplace=True,
                ),
            ]
        )
        image = transform(image)
        if self._transform:
            image = self._transform(image)
        return image.to(torch.float32), image_label.to(torch.float32)