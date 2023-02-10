# Python imports
from typing import Optional, Union

# Third party imports
import torch

# Self imports

class ResNet(torch.nn.Module):
    """
    Output of Conv2D can be calculated as:
        output_image_size =  [{(input_image_height_or_weight - karnel_size) + 2 * padding} / stride] +1
    """

    def __init__(self):
        super(ResNet, self).__init__()
        # 1 image = 3 channel * 300 height * 300 width
        ## Conv layer
        self.conv_2d = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=0
            )
        self.batch_norm = torch.nn.BatchNorm2d(num_features=64)
        self.relu = torch.nn.ReLU()
        self.max_pooling= torch.nn.MaxPool2d(kernel_size=3, stride=2)

        ## Res block 1
        self.res_1_conv_2d_1 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
            )
        self.res_1_batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)
        self.res_1_relu_1 = torch.nn.ReLU()
        self.res_1_conv_2d_2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
            )
        self.res_1_batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        self.res_1_relu_2 = torch.nn.ReLU()

        # Res block 2
        self.res_2_downsample = res = self.get_downsample(64, 128)
        self.res_2_conv_2d_1 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.res_2_batch_norm_1 = torch.nn.BatchNorm2d(num_features=128)
        self.res_2_relu_1 = torch.nn.ReLU()
        self.res_2_conv_2d_2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            )
        self.res_2_batch_norm_2 = torch.nn.BatchNorm2d(num_features=128)
        self.res_2_relu_2 = torch.nn.ReLU()

        # Res block 3
        self.res_3_downsample = res = self.get_downsample(128, 256)
        self.res_3_conv_2d_1 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.res_3_batch_norm_1 = torch.nn.BatchNorm2d(num_features=256)
        self.res_3_relu_1 = torch.nn.ReLU()
        self.res_3_conv_2d_2 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            )
        self.res_3_batch_norm_2 = torch.nn.BatchNorm2d(num_features=256)
        self.res_3_relu_2 = torch.nn.ReLU()

        # Res block 4
        self.res_4_downsample = res = self.get_downsample(256, 512)
        self.res_4_conv_2d_1 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.res_4_batch_norm_1 = torch.nn.BatchNorm2d(num_features=512)
        self.res_4_relu_1 = torch.nn.ReLU()
        self.res_4_conv_2d_2 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            )
        self.res_4_batch_norm_2 = torch.nn.BatchNorm2d(num_features=512)
        self.res_4_relu_2 = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten(start_dim=1)

        self.fully_connected = torch.nn.Linear(in_features=512, out_features=2)

        self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        ## Res 1
        res = x
        x = self.res_1_conv_2d_1(x)
        x = self.res_1_batch_norm_1(x)
        x = self.res_1_relu_1(x)
        x = self.res_1_conv_2d_2(x)
        x = self.res_1_batch_norm_2(x) + res
        x = self.res_1_relu_2(x)
        
        # Res 2
        res = self.res_2_downsample(x)
        x = self.res_2_conv_2d_1(x)
        x = self.res_2_batch_norm_1(x)
        x = self.res_2_relu_1(x)
        x = self.res_2_conv_2d_2(x)
        x = self.res_2_batch_norm_2(x) + res
        x = self.res_2_relu_2(x)

        # Res 3
        res = self.res_3_downsample(x)
        x = self.res_3_conv_2d_1(x)
        x = self.res_3_batch_norm_1(x)
        x = self.res_3_relu_1(x)
        x = self.res_3_conv_2d_2(x)
        x = self.res_3_batch_norm_2(x) + res
        x = self.res_3_relu_2(x)

        # Res 4
        res = self.res_4_downsample(x)
        x = self.res_4_conv_2d_1(x)
        x = self.res_4_batch_norm_1(x)
        x = self.res_4_relu_1(x)
        x = self.res_4_conv_2d_2(x)
        x = self.res_4_batch_norm_2(x) + res
        x = self.res_4_relu_2(x)
        x = torch.mean(x.view(x.size(0), x.size(0), -1), dim=2)
        x =  self.flatten(x)
        x =  self.fully_connected(x)
        x =  self.sigmoid(x)
        return x

    def get_downsample(self, in_channels, out_channels):
        modules = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        return modules