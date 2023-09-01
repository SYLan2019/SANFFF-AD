#!/usr/bin/python
# author mawei
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import config as c
from model import *

import utils
def apply_mask(image, mask, color, alpha=0):#?
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    #     在遮掩处将图像遮掩，用alpha表示透明度，并用color在遮掩处给上新值，即颜色区分出来
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# image mask 原图片和attnmask矩阵
def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    #
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()#?

    N = 1
    # 增加了一个维度
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))#图像进行模糊处理
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        contour = False
        if contour:#找到掩膜的边界，将创建多面性对象将掩膜的边界在轴上画出来
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask#?
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor=color, edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

os.environ['TORCH_HOME'] = 'models\\EfficientNet'
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='checkpoint/dino_deitsmall8_300ep_pretrain.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default='data/hazelnut/test/crack/000.png', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(384, 384), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.5, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model or build flow model
    model = load_model(c.modelname)
    model.eval()
    net = model.nf_mlp.module_list
    model_ad = ADwithGlow()
    feature_extractor = model_ad.feature_extractor
    with open(args.image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    transform = pth_transforms.Compose([transforms.Resize(c.img_size, Image.ANTIALIAS),
               transforms.CenterCrop(c.img_size),
               transforms.ToTensor(),
               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
       ])
    img = transform(img)

    # make the image divisible by the patch size
    # 3,384,384
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    # batchsize 维度
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size
    inp = list()
    img_inp = feature_extractor(img)
    img_inp = img_inp.transpose(1,2)
    inp.append(img_inp)
    for i in range(9):
        if i==0:
            continue
        if i ==8:
            inp,attentions = net[i](inp, attention=True)
        else:
            inp = net[i](inp)

    nh = 8
    attentions=attentions[:,:,0,1:].reshape(8,-1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)#第i位为到该位的和
        th_attn = cumval > (1 - args.threshold)#bool矩阵， True代表从初始位到该位置的和大于 1-0.9
        # 获取当前数组中每一个数在原数组中的位置
        idx2 = torch.argsort(idx)
        for head in range(nh):
            # 将bool矩阵按原attn重新归位
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate 等比例缩放
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    #4,24,24
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].detach().cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
