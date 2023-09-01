#!/usr/bin/python
# author mawei
import numpy as np
from torch import nn
from torch import Tensor, LongTensor
from torch.nn import functional as F
import torch
import math
from typing import Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2,
        #                    groups=dim)
        # self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2,
        #                    groups=dim)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, 2 * dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention=False):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # H = int((N-1)**(0.5))
        # x_spa = x[:, 1:].reshape(B, C, H, H)
        # k = self.k(x_spa).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q = q * self.scale
        # v = self.v(x_spa).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        attention = attention
        if attention == True:
            return x_cls, attn

        return x_cls


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, 2 * dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention=False):
        B, N, C = x.shape
        q = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)[:,1:]
        attention = attention
        if attention == True:
            return out, attn

        return out



# class HiLo(nn.Module):
#     """
#     HiLo Attention
#     Paper: Fast Vision Transformers with HiLo Attention
#     Link: https://arxiv.org/abs/2205.13213
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=3, alpha=0.5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         head_dim = int(dim/num_heads)
#         self.dim = dim
#
#         # self-attention heads in Lo-Fi
#         self.l_heads = int(num_heads * alpha)
#         # token dimension in Lo-Fi
#         self.l_dim = self.l_heads * head_dim
#
#         # self-attention heads in Hi-Fi
#         self.h_heads = num_heads - self.l_heads
#         # token dimension in Hi-Fi
#         self.h_dim = self.h_heads * head_dim
#
#         # local window size. The `s` in our paper.
#         self.ws = window_size
#
#         if self.ws == 1:
#             # ws == 1 is equal to a standard multi-head self-attention
#             self.h_heads = 0
#             self.h_dim = 0
#             self.l_heads = num_heads
#             self.l_dim = dim
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # Low frequence attention (Lo-Fi)
#         if self.l_heads > 0:
#             if self.ws != 1:
#                 self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
#             self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
#             self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
#             self.l_proj = nn.Linear(self.l_dim, self.l_dim )
#
#         # High frequence attention (Hi-Fi)
#         if self.h_heads > 0:
#             self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
#             self.h_proj = nn.Linear(self.h_dim, self.h_dim )
#
#     def hifi(self, x):
#         B, H, W, C = x.shape
#         h_group, w_group = H // self.ws, W // self.ws
#
#         total_groups = h_group * w_group
#
#         x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
#
#         qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
#         attn = attn.softmax(dim=-1)
#         attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
#         x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
#
#         x = self.h_proj(x)
#         return x
#
#     def lofi(self, x):
#         B, H, W, C = x.shape
#
#         q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
#
#         if self.ws > 1:
#             x_ = x.permute(0, 3, 1, 2)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
#         x = self.l_proj(x)
#         return x
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#
#         x = x.reshape(B, H, W, C)
#
#         if self.h_heads == 0:
#             x = self.lofi(x)
#             return x.reshape(B, N, C)
#
#         if self.l_heads == 0:
#             x = self.hifi(x)
#             return x.reshape(B, N, C)
#
#         hifi_out = self.hifi(x)
#         lofi_out = self.lofi(x)
#
#         x = torch.cat((hifi_out, lofi_out), dim=-1)
#         x = x.reshape(B, N, C)
#
#         return x
#
#     def flops(self, H, W):
#         # pad the feature map when the height and width cannot be divided by window size
#         Hp = self.ws * math.ceil(H / self.ws)
#         Wp = self.ws * math.ceil(W / self.ws)
#
#         Np = Hp * Wp
#
#         # For Hi-Fi
#         # qkv
#         hifi_flops = Np * self.dim * self.h_dim * 3
#         nW = (Hp // self.ws) * (Wp // self.ws)
#         window_len = self.ws * self.ws
#         # q @ k and attn @ v
#         window_flops = window_len * window_len * self.h_dim * 2
#         hifi_flops += nW * window_flops
#         # projection
#         hifi_flops += Np * self.h_dim * self.h_dim
#
#         # for Lo-Fi
#         # q
#         lofi_flops = Np * self.dim * self.l_dim
#         kv_len = (Hp // self.ws) * (Wp // self.ws)
#         # k, v
#         lofi_flops += kv_len * self.dim * self.l_dim * 2
#         # q @ k and attn @ v
#         lofi_flops += Np * self.l_dim * kv_len * 2
#         # projection
#         lofi_flops += Np * self.l_dim * self.l_dim
#
#         return hifi_flops + lofi_flops


# def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
#     """
#     Args:
#         x: BCHW tensor
#         region size: int
#         num_heads: number of attention heads
#     Return:
#         out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
#         region_h, region_w: number of regions per col/row
#     """
#     B, C, H, W = x.size()
#     region_h, region_w = H // region_size[0], W // region_size[1]
#     x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
#     # (2,8,64,3,8,3,8) --->(2,8,8,8,3,3,64)  --->(2,8,64,9,64)
#     x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
#     return x, region_h, region_w
#
#
# def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
#     """
#     Args:
#         x: (bs, nhead, nregion, reg_size^2, head_dim)
#     Return:
#         x: (bs, C, H, W)
#     """
#     bs, nhead, nregion, reg_size_square, head_dim = x.size()
#     x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
#     x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
#                                                     region_h * region_size[0], region_w * region_size[1])
#     return x
#
#
# def regional_routing_attention_torch(
#         query: Tensor, key: Tensor, value: Tensor, scale: float,
#         region_graph: LongTensor, region_size: Tuple[int],
#         kv_region_size: Optional[Tuple[int]] = None,
#         auto_pad=True) -> Tensor:
#     """
#     Args:
#         query, key, value: (B, C, H, W) tensor
#         scale: the scale/temperature for dot product attention
#         region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
#         region_size: region/window size for queries, (rh, rw)
#         key_region_size: optional, if None, key_region_size=region_size
#         auto_pad: required to be true if the input sizes are not divisible by the region_size
#     Return:
#         output: (B, C, H, W) tensor
#         attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
#     """
#     kv_region_size = kv_region_size or region_size
#     bs, nhead, q_nregion, topk = region_graph.size()
#
#     # Auto pad to deal with any input size
#     q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
#     if auto_pad:
#         _, _, Hq, Wq = query.size()
#         q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
#         q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
#         if (q_pad_b > 0 or q_pad_r > 0):
#             query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding
#
#         _, _, Hk, Wk = key.size()
#         kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
#         kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
#         if (kv_pad_r > 0 or kv_pad_b > 0):
#             key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
#             value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
#
#     # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
#     # 2,8,8*8,3*3,64
#     query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
#     key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
#     value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)
#
#     # gather key and values.
#     # TODO: is seperate gathering slower than fused one (our old version) ?
#     # torch.gather does not support broadcasting, hence we do it manually
#     bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
#     # 64个区域 每个区域最相关的4个部分 2,8,64,4  --- 2,8,64,4,9,64
#     broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
#         expand(-1, -1, -1, -1, kv_region_size, head_dim)
#     # 在区域表示部分扩展64倍 2,8,64,9,64 --- 2,8,1,64,9,64 ---2,8,64,64,9,64
#     key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
#                          expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
#                          index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
#     value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
#                            expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
#                            index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
#     # k_g : 2,8,64,4,9,64  -- q:2,8,64,9,64  @ k_g:(2,8,64,36,64--2,8,64,64,36)  = 2,8,64,9,36
#     # token-to-token attention
#     # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
#     # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
#     # TODO: mask padding region
#     attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
#     attn = torch.softmax(attn, dim=-1)
#     # attn(2,8,64,9,36) @ k_g:(2,8,64,36,64) = 2,8,64,9,64
#     # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
#     # -> (bs, nhead, q_nregion, reg_size, head_dim)
#     output = attn @ value_g.flatten(-3, -2)
#
#     # to BCHW format 2,8,64,9,64 -->2,512,24,24
#     output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)
#
#     # remove paddings if needed
#     if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
#         output = output[:, :, :Hq, :Wq]
#
#     return output, attn
#
# class nchwAttentionLePE(nn.Module):
#     """
#     Attention with LePE, takes nchw input
#     """
#
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=3):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = qk_scale or self.head_dim ** -0.5
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
#                               groups=dim) if side_dwconv > 0 else \
#             lambda x: torch.zeros_like(x)
#
#     def forward(self, x: torch.Tensor):
#         """
#         args:
#             x: NCHW tensor
#         return:
#             NCHW tensor
#         """
#         B, C, H, W = x.size()
#         q, k, v = self.qkv.forward(x).chunk(3, dim=1)  # B, C, H, W
#
#         attn = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2) @ \
#                k.view(B, self.num_heads, self.head_dim, H * W)
#         attn = torch.softmax(attn * self.scale, dim=-1)
#         attn = self.attn_drop(attn)
#
#         # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
#         output: torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
#         output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         output = output + self.lepe(v)
#
#         output = self.proj_drop(self.proj(output))
#
#         return output
#
# class nchwBRA(nn.Module):
#     """Bi-Level Routing Attention that takes nchw input
#     Compared to legacy version, this implementation:
#     * removes unused args and components
#     * uses nchw input format to avoid frequent permutation
#     When the size of inputs is not divisible by the region size, there is also a numerical difference
#     than legacy implementation, due to:
#     * different way to pad the input feature map (padding after linear projection)
#     * different pooling behavior (count_include_pad=False)
#     Current implementation is more reasonable, hence we do not keep backward numerical compatiability
#     """
#
#     def __init__(self, dim, num_heads=8, n_win=8, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False,
#                  attn_backend='torch'):
#         super().__init__()
#         # local attention setting
#         self.dim = dim
#         self.num_heads = num_heads
#         assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
#         self.head_dim = self.dim // self.num_heads
#         self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.
#
#         ################side_dwconv (i.e. LCE in Shunted Transformer)###########
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
#                               groups=dim) if side_dwconv > 0 else \
#             lambda x: torch.zeros_like(x)
#
#         ################ regional routing setting #################
#         self.topk = topk
#         self.n_win = n_win  # number of windows per row/col
#
#         ##########################################
#
#         self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
#         self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)
#
#         if attn_backend == 'torch':
#             self.attn_fn = regional_routing_attention_torch
#         else:
#             raise ValueError('CUDA implementation is not available yet. Please stay tuned.')
#
#     def forward(self, x: Tensor, ret_attn_mask=False):
#         """
#         Args:
#             x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
#         Return:
#             NCHW tensor
#         """
#         N, C, H, W = x.size()
#         region_size = (H // self.n_win, W // self.n_win)
#
#         # STEP 1: linear projection
#         qkv = self.qkv_linear.forward(x)  # ncHW
#         q, k, v = qkv.chunk(3, dim=1)  # ncHW
#
#         # STEP 2: region-to-region routing
#         # NOTE: ceil_mode=True, count_include_pad=False = auto padding
#         # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
#         # 2,512,8,8-->2,8,8,512-->2,64,512 k_r:2,512,64  a_r = 2,64,64
#         q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
#         k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
#         q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
#         k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
#         a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
#         # 没个区域最相关的qiantopk个元素 2,64,4 -- 2,8,64,4
#         _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
#         idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)
#
#         # STEP 3: token to token attention (non-parametric function)
#         output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
#                                         region_graph=idx_r, region_size=region_size
#                                         )
#
#         output = output + self.lepe(v)  # ncHW
#         output = self.output_linear(output)  # ncHW
#
#         if ret_attn_mask:
#             return output, attn_mat
#
#         return output

if __name__ == '__main__':
    x = torch.randn((2, 512))
    x_spa = torch.randn((2, 512, 24, 24))
    # net = HiLo(dim=384,num_heads=8,window_size=3,alpha=1)
    net = Class_Attention_Conv(dim=512, num_heads=4)
    y = net(x, x_spa)

    # alpha 调节的是 lofi的比例，alpha =0 纯窗口注意力机制 alpha = 1
    # print(y.shape)
