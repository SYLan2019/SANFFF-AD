import math

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from efficient_subnet import *
import timm
from efficientnet_pytorch import EfficientNet

import clip
import config as c
from freia_funcs_oral import *
from pytorch_pretrained_vit.model import ViT

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head_conv(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        # nodes.append(Node([nodes[-1].out0], Norm, {}, name=F'norm_{k}'))
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # nodes.append(Node([nodes[-1].out0], Invconv, {}, name=F'inconv_{k}'))
        if k % 2 == 0:
            kernel = 3
        else:
            kernel = 3
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': F_conv,
                           'F_args': {'channels_hidden': c.fc_internal, 'kernel_size': kernel,'subnet':c.subnet}},
                          name=F'fc_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


def nf_head_mlp(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        # nodes.append(Node([nodes[-1].out0], Norm, {}, name=F'norm_{k}'))
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': F_fully_connected,
                           'F_args': {'channels_hidden': c.fc_internal}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class ADwithGlow(nn.Module):
    def __init__(self):
        super(ADwithGlow, self).__init__()
        if c.extractor == 'cait':
            self.feature_extractor = timm.create_model(
                'cait_m48_448',
                pretrained=True,
            )
            self.feature_extractor.head = nn.Identity()
        elif c.extractor == 'deit':
            self.feature_extractor = timm.create_model(
                'deit_base_distilled_patch16_384',
                pretrained=True
            )
        elif c.extractor == 'VIT':
            self.feature_extractor = ViT('B_16_imagenet1k', pretrained=True)
            self.feature_extractor.fc = nn.Identity()
        elif c.extractor == 'clip':
            model, preprocess = clip.load('ViT-B/32', download_root='model/')
            self.model = model
            self.feature_extractor = model.visual
        elif c.extractor == 'effnetB5':
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.nf_mlp = nf_head_mlp()
        # self.pool = nn.AdaptiveAvgPool2d((1,1))
    def eff_ext(self, x, use_layer=36):#38可以尝试一下
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x
    def cait_ext(self,x):
        x = self.feature_extractor.patch_embed(x)
        x = x + self.feature_extractor.pos_embed
        x = self.feature_extractor.pos_drop(x)
        for i in range(41):  # paper Table 6. Block Index = 40
            x = self.feature_extractor.blocks[i](x)
        N, _, C = x.shape
        x = self.feature_extractor.norm(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
        return x
    def deit_ext(self,x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        if self.feature_extractor.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        for i in range(8):  # paper Table 6. Block Index = 7
            x = self.feature_extractor.blocks[i](x)
        x = self.feature_extractor.norm(x)
        x1 = x[:,0,:].unsqueeze(1)
        x = x[:, 2:, :]
        x = torch.cat((x1,x),dim=1)
        # N, _, C = x.shape
        # x = x.permute(0, 2, 1)
        # x = x.reshape(N, C, c.img_size[0] // 16, c.img_size[1] // 16)
        return x
    def forward(self, x):

        if c.pretrained:
            # B,N,C ---> B,C,N
            # x = x.transpose(1,2)
            z = self.nf_mlp(x)
            # B.C.N---B,N,C
            return z
        else:
            if c.extractor == 'cait':
                feat_sem = self.feature_extractor(x)
            elif c.extractor == 'vit_base':
                feat_sem = self.feature_extractor(x)
            else:
                feat_sem = self.feature_extractor(x)
            feat_sem = feat_sem.transpose(1, 2)
            z_sem = self.nf_mlp(feat_sem)
            return z_sem


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
    # model2, preprocess = clip.load('ViT-B/32',download_root='model/')
    # x=torch.randn((32,3,224,224))
    # net = model2.visual
    # z2 = net(x)
    # z1 = model2.encode_text(clip.tokenize('A photo of cat'))
    # z1 = z1.expand((c.batch_size,-1))
    # z = torch.cat((z1,z2),dim=1)
    # model1 = model2.visual
    # model1.proj = None
    # print(summary(model, (3, 384, 384), device=c.device))
    # # from torchsummary import summary
    # # summary(model,(3,64,64)
    # model = ViT('B_16_imagenet1k', pretrained=True)
    # model.fc = nn.Identity()
    # model = ADwithGlow()
    # nf_conv = nf_head_conv()
    # z1,z2 = model(x)
    # loss1 = get_loss(z1, model.norms[0].jacobian(run_forward=False))
    # loss2 = get_loss(z2,model.norms[1].jacobian(run_forward=False))
    # y = model(x)
    # # z2 = nf_conv(y2)
    # print(z1.shape)

    import thop
    # model1 = nf_head_conv()
    model2 = ADwithGlow()
    x1 = torch.randn((2, 577,768))
    y = model2(x1)
    loss2 = get_loss(y, model2.nf_mlp.jacobian(run_forward=False))
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
    # model = load_model(c.modelname)


    print(1)
