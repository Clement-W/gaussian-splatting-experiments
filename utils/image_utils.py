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

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0,  1, 0], [1, -4, 1], [0,  1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_sobel(image):
    SOBEL_X.to(image.device)
    SOBEL_Y.to(image.device)
    gray = (0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]).unsqueeze(0)
    edge_x = F.conv2d(gray, SOBEL_X, padding=1)
    edge_y = F.conv2d(gray, SOBEL_Y, padding=1)
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge_map

def compute_laplacian(image):
    LAPLACIAN.to(image.device)
    gray = (0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]).unsqueeze(0)
    return F.conv2d(gray, LAPLACIAN, padding=1)