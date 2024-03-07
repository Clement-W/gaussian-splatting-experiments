#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

class GlobalKernels:
    def __init__(self):
        self.SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def sobel_to_device(self,device):
        self.SOBEL_X = self.SOBEL_X.to(device)
        self.SOBEL_Y = self.SOBEL_Y.to(device)
    
    def laplacian_to_device(self, device):
        self.LAPLACIAN = self.LAPLACIAN.to(device)

kernels = GlobalKernels()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_sobel(image):
    kernels.sobel_to_device(image.device)
    gray = (0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]).unsqueeze(0)
    edge_x = F.conv2d(gray, kernels.SOBEL_X, padding=1)
    edge_y = F.conv2d(gray, kernels.SOBEL_Y, padding=1)
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge_map

def compute_laplacian(image):
    kernels.laplacian_to_device(image.device)
    gray = (0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]).unsqueeze(0)
    return F.conv2d(gray, kernels.LAPLACIAN, padding=1)