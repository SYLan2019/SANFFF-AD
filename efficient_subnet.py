# #!/usr/bin/python
# # author mawei
import math

import torch
import torch.nn as nn
from einops import rearrange
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F

from typing import Optional, Tuple


def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    # (2,8,64,8,3,8,3) --->(2,8,8,8,3,3,64)  --->(2,8,64,9,64)
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w

# Todo attention model replace attention function


def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int],
        kv_region_size: Optional[Tuple[int]] = None,
        ) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """

    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    # 64个区域 每个区域最相关的4个部分 2,8,64,4  --- 2,8,64,4,9,64
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    # 在区域表示部分扩展64倍 2,8,64,9,64 --- 2,8,1,64,9,64 ---2,8,64,64,9,64
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    # k_g : 2,8,64,4,9,64  -- q:2,8,64,9,64  @ k_g:(2,8,64,36,64--2,8,64,64,36)  = 2,8,64,9,36
    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query.unsqueeze(2) * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # attn(2,8,64,9,36) @ k_g:(2,8,64,36,64) = 2,8,64,9,64
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)
    # to BCHW format 2,8,64,9,64 -->2,512,24,24
    # output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)
    output = output.view(query.size(0), -1)
    return output, attn

class BRA(nn.Module):
    """Bi-Level Routing Attention that takes nchw input
    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation
    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)
    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """

    def __init__(self, dim, num_heads=8, n_win=8, qk_scale=None, topk=32, side_dwconv=3, auto_pad=False,
                 attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        # self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
        #                       groups=dim) if side_dwconv > 0 else \
        #     lambda x: torch.zeros_like(x)

        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################
        self.q = nn.Linear(self.dim, self.dim)
        self.kv_linear = nn.Conv2d(self.dim, 2 * self.dim, kernel_size=1)
        self.output_linear = nn.Linear(self.dim, 2 * self.dim)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N - 1))
        region_size = (H // self.n_win, W // self.n_win)
        x_spa = x[:, 1:].reshape(B, C, H, W)
        # STEP 1: linear projection
        q = self.q(x[:, 0]).unsqueeze(1)
        kv = self.kv_linear.forward(x_spa)  # ncHW
        k, v = kv.chunk(2, dim=1)  # ncHW

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        # 2,512,8,8-->2,8,8,512-->2,64,512 k_r:2,512,64  a_r = 2,64,64
        # k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        k_r = F.max_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        # q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q @ k_r  # n(hw)(hw), adj matrix of regional graph
        # 没个区域最相关的qiantopk个元素 2,64,4 -- 2,8,64,4
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token to token attention (non-parametric function)
        q = q.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                        )

        # output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output

from subnet import *
import thop
if __name__ == '__main__':
    model1 = BRA(dim=512,topk=64,num_heads=8)
    model2 = Class_Attention(dim=512,num_heads=8)
    x = torch.randn((2, 577, 512))
    y = model1(x)
    flops1, params1 = thop.profile(model1, inputs=(x,))
    flops1, params1 = thop.clever_format([flops1, params1], )
    flops2, params2 = thop.profile(model2, inputs=(x,))
    flops2, params2 = thop.clever_format([flops2, params2], )
    print(1)
