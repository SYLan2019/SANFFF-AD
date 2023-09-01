import math

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from subnet import *
import timm
from efficientnet_pytorch import EfficientNet

import clip
import config as c
# from freia_funcs import *
from freia_funcs_reverse import *
from pytorch_pretrained_vit.model import ViT
from model import *
from feature_aggrator import *

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


# def nf_head_conv(input_dim=c.n_feat):
#     nodes = list()
#     nodes.append(InputNode(input_dim, name='input'))
#     for k in range(c.n_coupling_blocks):
#         # nodes.append(Node([nodes[-1].out0], Norm, {}, name=F'norm_{k}'))
#         nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
#         # nodes.append(Node([nodes[-1].out0], Invconv, {}, name=F'inconv_{k}'))
#         if k % 2 == 0:
#             kernel = 3
#         else:
#             kernel = 3
#         nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
#                           {'clamp': c.clamp, 'F_class': F_conv,
#                            'F_args': {'channels_hidden': c.fc_internal, 'kernel_size': kernel, 'subnet': c.subnet}},
#                           name=F'fc_{k}'))
#
#     nodes.append(OutputNode([nodes[-1].out0], name='output'))
#     coder = ReversibleGraphNet(nodes)
#     return coder
#
#
# def nf_head_mlp(input_dim=c.n_feat):
#     nodes = list()
#     nodes.append(InputNode(input_dim, name='input'))
#     for k in range(c.n_coupling_blocks):
#         # nodes.append(Node([nodes[-1].out0], Norm, {}, name=F'norm_{k}'))
#         nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
#         nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
#                           {'clamp': c.clamp, 'F_class': F_fully_connected,
#                            'F_args': {'channels_hidden': c.fc_internal}},
#                           name=F'fc_{k}'))
#     nodes.append(OutputNode([nodes[-1].out0], name='output'))
#     coder = ReversibleGraphNet(nodes)
#     return coder

def nf_head_attention(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': F_attention,
                           'F_args': {'channels_hidden': c.fc_internal}},
                          name=F'fc_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class ADwithGlow(nn.Module):
    def __init__(self):
        super(ADwithGlow, self).__init__()
        if c.extractor == 'DINO':
            self.feature_extractor = timm.create_model(model_name='vit_base_patch8_224_dino', pretrained=True,
                                                       checkpoint_path='models\EfficientNet\hub\checkpoints\dino_vitbase8_pretrain.pth')
            self.feature_extractor.head = nn.Identity()
        elif c.extractor == 'VIT':
            self.feature_extractor = ViT('L_16_imagenet1k', pretrained=True)
            self.feature_extractor.fc = nn.Identity()
        elif c.extractor == 'clip':
            model, preprocess = clip.load('ViT-L/14@336px', download_root='model/')
            self.model = model
            self.feature_extractor = model.visual
            self.feature_extractor.proj = None
        elif c.extractor == 'DINOV2':
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        # self.feature_aggreator = FeatureAggrator(input_dim=c.n_feat)

        # self.nf_mlp = nf_head_mlp()
        self.nf_mlp = nf_head_attention()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

    def dino_ext(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.norm_pre(x)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x

    def vit_ext(self, x):
        # B, N, C
        # fem = self.feature_extractor(x)
        fem = x
        x_1 = fem[:, 0, :]
        x_2 = fem[:, 1:, :]
        B, N, C = x_2.shape
        H = W = int(N ** 0.5)
        # 24 x 24
        x_2 = x_2.reshape(B, C, H, W)
        #  C 8 x 8 --- C 64  -- 64 C
        x_2 = self.pool(x_2).reshape(B, C, -1).permute(0, 2, 1)
        x = torch.cat((x_1.unsqueeze(1), x_2), dim=1)
        return x

    def dinov2_ext(self, x):
        x = self.feature_extractor.prepare_tokens_with_masks(x, masks=None)
        for blk in self.feature_extractor.blocks:
            x = blk(x)
        x_norm = self.feature_extractor.norm(x)
        return x_norm

    def efficient_extract(self, x):
        x_sem = x[:, 0]
        x_spa = x[:, 1:]
        x_spa = self.feature_aggreator(x_spa)
        out = torch.cat((x_sem.unsqueeze(1), x_spa), dim=1)
        return out

    def forward(self, x):
        # if c.pretrained:
        #     # B,N,C ---> B,C,N
        #     x = x.transpose(1, 2)
        #     z_sem = self.nf_mlp(x)
        #     z = z_sem.transpose(1, 2)[:, 0]
        #     return z
        # else:
        #     raise AttributeError
        if c.pretrained:
            # B,N,C ---> B,C,N
            x = x.transpose(1, 2)
            z = self.nf_mlp(x)
            z = z.transpose(1, 2)[:, 1:]
            return z
        else:
            raise AttributeError


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    # # print(summary(model.feature_extractor,(3,224,224),device=c.device))
    os.environ['TORCH_HOME'] = 'models\\EfficientNet'
    # vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    # from torchsummary import summary
    # summary(model,(3,64,64)
    # model = ViT('B_16_imagenet1k', pretrained=True)
    # model.fc = nn.Identity()
    # model = ADwithGlow()
    # x = torch.randn((2,3,224,224))
    # y = model.dino_ext(x)
    # model = nf_head_mlp()
    x = torch.randn((2, 577, 1024))
    # y = model(x)
    # import thop
    # flops, params = thop.profile(model, inputs=(x,))
    # flops, params = thop.clever_format([flops, params], )
    # net = model.nf_mlp.module_list
    model = ADwithGlow()
    y = model(x)
    loss = get_loss(y, model.nf_mlp.jacobian(run_forward=False))
    # x = torch.randn((2, 1024, 577))
    # inp = list()
    #
    # inp.append(x)
    # y = model(x)
    # loss = get_loss(y, model.nf_mlp.jacobian(run_forward=False))
    # for i in range(9):
    #     if i == 0:
    #         continue
    #     if i == 8:
    #         inp, attention = net[i](inp, attention=True)
    #     else:
    #         inp = net[i](inp)

    # model1 = nf_head_conv()
    # x2 = torch.randn(1, 768)
    # y = model1(x1)
    # flops, params = thop.profile(model1, inputs=(x1,))
    # flops, params = thop.clever_format([flops, params], )
    # flops2, params2 = thop.profile(model2, inputs=(x2,))
    # flops2, params2 = thop.clever_format([flops2, params2], )
    # flops1,parmas1 = get_model_complexity_info(model1,(768,24,24),as_strings=True, print_per_layer_stat=True)
    # flops2, parmas2 = get_model_complexity_info(model2,(768,),as_strings=True, print_per_layer_stat=True)
    # summary(model1,(768,224,224))
    # print(flops, params)
    # print(flops2, params2)
    print(1)
