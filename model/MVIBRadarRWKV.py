import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torch.utils.cpp_extension import load
import math
from timm.models.layers import trunc_normal_
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"  # 启用详细日志
# os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"  # 切换临时目录

print("start loading cuda kernel")
wkv_cuda = load(name="bi_wkv", sources=["./RadarRWKV/cuda_new/bi_wkv.cpp", 
                                        "./RadarRWKV/cuda_new/bi_wkv_kernel.cu"],
                verbose=2, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '-allow-unsupported-compiler', 
                                              '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])

print("end loading")
NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)
        

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor



def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    # B, N, C = input.shape
    # input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    if gamma == 1/4:
        
        output = torch.zeros_like(input)
        output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
        output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
        output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
        output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
        output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    
    else:
        output = torch.zeros_like(input)
        for i in range(int(C*gamma*8)):
            output[:, i, :, shift_pixel+i:W] = input[:, i, :, 0:W-shift_pixel-i]
        for i in range(int(C*gamma*8), int(2*C*gamma*8)):
            output[:, i, :, 0:W-shift_pixel-i] = input[:, i, :, shift_pixel+i:W]
        for i in range(int(2*C*gamma*8), int(3*C*gamma*8)):
            output[:, i, shift_pixel+i:H, :] = input[:, i, 0:H-shift_pixel-i, :]
        for i in range(int(3*C*gamma*8), int(4*C*gamma*8)):
            output[:, i, 0:H-shift_pixel-i, :] = input[:, i, shift_pixel+i:H, :]
        output[:, int(C*gamma*4*8):, ...] = input[:, int(C*gamma*4*8):, ...]

        # return output.flatten(2).transpose(1, 2)
    return output


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(in_channels=gating_channels, out_channels=inter_channels, kernel_size=1)
        
        self.W_x = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
        
        self.psi = nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=1)
        self.sigm = nn.Sigmoid()
        
        self.relu = nn.ReLU()
        # self.relu = nn.GELU()

    def forward(self, x, g):
        g1 = self.W_g(g).contiguous()
        x1 = self.W_x(x).contiguous()
        psi = self.relu(g1 + x1)
        psi = self.sigm(self.psi(psi))

        # output_real = torch.mul(x.real, psi.real) - torch.mul(x.imag, psi.imag)
        # output_imag = torch.mul(x.real, psi.imag) + torch.mul(x.imag, psi.real)
        return torch.mul(x, psi)



class SpatialMix(nn.Module):
    def __init__(self, dim, layer_id, n_layer, num_heads, adapkc, init_mode='global', shift_mode='q_shift'):
        super().__init__()
        self.dim = dim
        attn_dim = dim
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.number_heads = num_heads
        self.adapkc = adapkc

        self._init_weights(init_mode)

        self.shift = eval(shift_mode)
        
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        # self.decay = nn.Parameter(torch.randn((self.dim,)))
        # self.boost = nn.Parameter(torch.randn((self.dim,)))

        if adapkc:
            self.double_adapkc_block_1 = DoubleAdaPKC2D(in_ch=dim, out_ch=dim,
                                                 bias=True,
                                                 rb=1,
                                                 gb_max=3)
            print('adapkc')
            self.double_adapkc_block_2 = DoubleAdaPKC2D(in_ch=dim, out_ch=dim,
                                                 bias=True,
                                                 rb=1,
                                                 gb_max=3)
            print('adapkc')

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.dim)
                for h in range(self.dim):
                    decay_speed[h] = -5 + 8 * (h / (self.dim-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.dim)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.dim) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.dim)
                for i in range(self.dim):
                    x[0, 0, i] = i / self.dim
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode == 'global':
            # global init
            self.spatial_decay = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_first = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
        elif init_mode == 'random':
            # random init
            self.spatial_decay = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_first = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_k = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_v = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_r = nn.Parameter(torch.randn((1, self.dim))) 
        else:
            raise NotImplementedError



    def jit_func(self, x, resolution):
        
        xx = rearrange(self.shift(x, gamma=1/4), "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        x = rearrange(x, "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        # if torch.isnan(xk).any() or torch.isinf(xk).any():
        #     raise ValueError("NaN or inf!")
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        x = x.permute(0,3,1,2)
        B, C, H, W = x.size()
        if self.adapkc:
            x = self.double_adapkc_block_1(x)

        sr, k, v = self.jit_func(x, resolution)

        # w 
        sr = sr.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 1, 3, 2, 4)
        sr = rearrange(sr, "b h n w d -> (b h) n w d")
        k = rearrange(k, "b h n w d -> (b h) n w d")
        v = rearrange(v, "b h n w d -> (b h) n w d")

        T = C // self.number_heads

        rwkv = []
        for i in range(self.number_heads):
            rwkv.append(RUN_CUDA(self.spatial_decay[i] / T, self.spatial_first[i] / T, k[:,i,:,:], v[:,i,:,:]).unsqueeze(1))
        
        rwkv = torch.cat(rwkv, 1)
        rwkv = sr * rwkv
        rwkv = rearrange(rwkv, "(b h) n w d -> b h n w d", b=B, h=H).permute(0, 3, 2, 1, 4)
        # rwkv = rearrange(rwkv, "b w n h d -> (b w) n h d")

        if self.adapkc:
            rwkv = self.double_adapkc_block_2(rwkv.permute(0,1,3,2,4).flatten(-2).permute(0,3,2,1))
            rwkv = rwkv.permute(0,2,3,1).view(B, H, W, self.number_heads, C//self.number_heads).permute(0,2,3,1,4)
        # h
        k = rearrange(k, "(b h) n w d -> b h n w d", b=B, h=H)
        v = rearrange(v, "(b h) n w d -> b h n w d", b=B, h=H)
        k = rearrange(k.permute(0, 3, 2, 1, 4), "b w n h d -> (b w) n h d")
        v = rearrange(v.permute(0, 3, 2, 1, 4), "b w n h d -> (b w) n h d")

        out = []
        for i in range(self.number_heads):
            out.append(RUN_CUDA(self.spatial_decay[i] / T, self.spatial_first[i] / T, k[:,i,:,:], v[:,i,:,:]).unsqueeze(1))
        out = rearrange(torch.cat(out, 1), "(b w) n h d -> b w n h d", b=B, w=W)
        out = rwkv * out
        out = out.permute(0, 3, 1, 2, 4).flatten(-2,-1)
        out = self.output(out)
        return out
    


class SpatialMixPeak(nn.Module):
    def __init__(self, dim, layer_id, n_layer, num_heads, init_mode='global', shift_mode='q_shift'):
        super().__init__()
        self.dim = dim
        attn_dim = dim
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.number_heads = num_heads

        self._init_weights(init_mode)

        self.shift = eval(shift_mode)
        
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        # self.decay = nn.Parameter(torch.randn((self.dim,)))
        # self.boost = nn.Parameter(torch.randn((self.dim,)))
        self.pk_conv = AdaPeak2D(dim, dim, bias=True, refer_band=1, gb_max=3)

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.dim)
                for h in range(self.dim):
                    decay_speed[h] = -5 + 8 * (h / (self.dim-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.dim)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.dim) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.dim)
                for i in range(self.dim):
                    x[0, 0, i] = i / self.dim
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode == 'global':
            # global init
            self.spatial_decay = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_first = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
        elif init_mode == 'random':
            # random init
            self.spatial_decay = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_first = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_k = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_v = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_r = nn.Parameter(torch.randn((1, self.dim))) 
        else:
            raise NotImplementedError



    def jit_func(self, x, resolution):

        x_pk = self.pk_conv(x)
        x_pk = rearrange(x_pk, "b c h w n -> (b n) c h w")
        xx = rearrange(self.shift(x_pk, gamma=1/4), "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        x = rearrange(x_pk, "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        
        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        # if torch.isnan(xk).any() or torch.isinf(xk).any():
        #     raise ValueError("NaN or inf!")

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, H, W, C = x.size()

        sr, k, v = self.jit_func(x.permute(0,3,1,2), resolution)

        sr = rearrange(sr, "(b n) (h w) c -> b (h w) n c", b=B, h=H, w=W) 
        sr = sr.contiguous().view(B*H*W, -1, self.number_heads, C//self.number_heads).permute(0, 2, 1, 3)
        k = rearrange(k, "(b n) (h w) c -> b (h w) n c", b=B, h=H, w=W) 
        k = k.contiguous().view(B*H*W, -1, self.number_heads, C//self.number_heads).permute(0, 2, 1, 3)
        v = rearrange(v, "(b n) (h w) c -> b (h w) n c", b=B, h=H, w=W) 
        v = v.contiguous().view(B*H*W, -1, self.number_heads, C//self.number_heads).permute(0, 2, 1, 3)

        T = C // self.number_heads

        rwkv = []
        for i in range(self.number_heads):
            rwkv.append(RUN_CUDA(self.spatial_decay[i] / T, self.spatial_first[i] / T, k[:,i,:,:], v[:,i,:,:]).unsqueeze(1))
        rwkv = torch.cat(rwkv, 1)

        # # 测试check循环和批处理结果一致
        # s = rwkv.view(B, H*W, self.number_heads, -1, C//self.number_heads)
        # temp = []
        # k_temp = k.view(B, H*W, self.number_heads, -1, C//self.number_heads)
        # v_temp = v.view(B, H*W, self.number_heads, -1, C//self.number_heads)
        # for i in range(H*W):
        #     t = []
        #     for j in range(self.number_heads):
        #         t.append(RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k_temp[:,i,j,:,:], v_temp[:,i,j,:,:]).unsqueeze(1))
        #     temp.append(torch.cat(t, 1))
        # temp = torch.stack(temp, 1)
        # print(torch.allclose(s, temp, atol=1e-6))

        
        rwkv = sr * rwkv
        rwkv = rearrange(rwkv, "(b h w) u n d -> b h w (u d) n", b=B, h=H, w=W, u=self.number_heads, d=C//self.number_heads)
        rwkv = rwkv.sum(-1)
        rwkv = self.output(rwkv)
        return rwkv
    


class SpatialMixAll(nn.Module):
    def __init__(self, dim, layer_id, n_layer, num_heads, adapkc, init_mode='global', shift_mode='q_shift'):
        super().__init__()
        self.dim = dim
        attn_dim = dim
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.number_heads = num_heads
        self.adapkc = adapkc

        self._init_weights(init_mode)

        self.shift = eval(shift_mode)
        
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        # self.decay = nn.Parameter(torch.randn((self.dim,)))
        # self.boost = nn.Parameter(torch.randn((self.dim,)))

        if adapkc:
            self.double_adapkc_block = DoubleAdaPKC2D(in_ch=dim, out_ch=dim,
                                                 bias=True,
                                                 rb=1,
                                                 gb_max=3)
            print('adapkc')

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.dim)
                for h in range(self.dim):
                    decay_speed[h] = -5 + 8 * (h / (self.dim-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.dim)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.dim) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.dim)
                for i in range(self.dim):
                    x[0, 0, i] = i / self.dim
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode == 'global':
            # global init
            self.spatial_decay = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_first = nn.Parameter(torch.zeros(self.dim)).reshape(self.number_heads, self.dim//self.number_heads)
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
        elif init_mode == 'random':
            # random init
            self.spatial_decay = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_first = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_k = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_v = nn.Parameter(torch.randn((1, self.dim))) 
            self.spatial_mix_r = nn.Parameter(torch.randn((1, self.dim))) 
        else:
            raise NotImplementedError



    def jit_func(self, x, resolution):
        
        xx = rearrange(self.shift(x, gamma=1/4), "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        x = rearrange(x, "b c h w -> b (h w) c", h=resolution[0], w=resolution[1])
        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        # if torch.isnan(xk).any() or torch.isinf(xk).any():
        #     raise ValueError("NaN or inf!")
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        x = x.permute(0,3,1,2)
        B, C, H, W = x.size()
        if self.adapkc:
            x = self.double_adapkc_block(x)

        sr, k, v = self.jit_func(x, resolution)

        # w 
        sr = sr.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 3, 1, 2, 4)
        k = k.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 3, 1, 2, 4)
        v = v.reshape(B, H, W, C).view(B, H, W, self.number_heads, C//self.number_heads).permute(0, 3, 1, 2, 4)
        sr = rearrange(sr, "b n h w d -> b n (h w) d")
        k = rearrange(k, "b n h w d -> b n (h w) d")
        v = rearrange(v, "b n h w d -> b n (h w) d")

        T = C // self.number_heads

        rwkv = []
        for i in range(self.number_heads):
            rwkv.append(RUN_CUDA(self.spatial_decay[i] / T, self.spatial_first[i] / T, k[:,i,:,:], v[:,i,:,:]).unsqueeze(1))
        
        rwkv = torch.cat(rwkv, 1)
        out = sr * rwkv
        out = out.permute(0, 2, 1, 3).flatten(-2,-1).reshape(B, H, W, C)
        out = self.output(out)
        return out
    

class ChannelMix(nn.Module):
    def __init__(self, dim, layer_id, n_layer, init_mode='global', shift_mode='q_shift', hidden_rate=4):
        super().__init__()
        self.dim = dim
        self.layer_id =layer_id
        self.n_layer = n_layer
        self._init_weights(init_mode)
        hidden_dim = int(hidden_rate * dim)

        self.shift = eval(shift_mode)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.dim)
                for i in range(self.dim):
                    x[0, 0, i] = i / self.dim
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'cross':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.dim]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.dim]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.dim]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, resolution):
        x = x.permute(0,3,1,2)
        xx = rearrange(self.shift(x, gamma=1/4), "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")

        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k.contiguous()))
        kv = self.value(k)
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv
    

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    

class RwkvBlock(nn.Module):
    def __init__(self, flag, layer_id, num_layer, dim, drop_path, num_heads: int, hidden_rate=4, with_ckpt=False, adaptive_pkc=False, peak_attn=False):
        super().__init__()
        self.layer_id = layer_id
        self.num_layer = num_layer
        self.with_ckpt = with_ckpt
        self.num_heads = num_heads
        self.adaptive_pkc = adaptive_pkc
        self.peak_attn = peak_attn

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if peak_attn:
            self.att = SpatialMixPeak(dim, layer_id, num_layer, num_heads)
        elif flag == 'chunk':
            self.att = SpatialMix(dim, layer_id, num_layer, num_heads, adapkc=adaptive_pkc)
        else:
            self.att = SpatialMixAll(dim, layer_id, num_layer, num_heads, adapkc=adaptive_pkc)
        self.ffn = ChannelMix(dim, layer_id, num_layer, hidden_rate=hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

        self.pos = DWConv2d(dim, 3, 1, 1)
        
        # if adaptive_pkc:
        #     self.pk_conv1 = AdaPeakConv2D(dim, dim, bias=True, refer_band=1, gb_max=3)
        #     self.bn1 = nn.BatchNorm2d(dim)
        #     self.act1 = nn.LeakyReLU(inplace=True)

    def _forward(self, x):
        resolution = (x.shape[1], x.shape[2])
        x = x + self.pos(x)
        x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), resolution))
        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), resolution).view(*x.shape))
        
        return x

    def forward(self, x):
        # if self.adaptive_pkc:
        #     x = self.pk_conv1(x)
        #     x = self.bn1(x)
        #     x = self.act1(x)
        if self.with_ckpt and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)
        

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class DetHeader(nn.Module):

    def __init__(self, use_bn=True, multiclass=False):
        super(DetHeader, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.multiclass = multiclass

        self.conv1 = conv3x3(128, 96, bias=bias, stride=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)      
        self.up = nn.ConvTranspose2d(96,96,3,2,1,1)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        if multiclass:
            self.clshead = conv3x3(96, 34*5, bias=True)
        else:
            self.clshead = conv3x3(96, 34, bias=True)     #34-->3D, 1-->2D

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(self.up(x))
        if self.use_bn:
            x = self.bn3(x)
        cls_out = self.clshead(x)
        if self.multiclass:
            B,C,H,W = cls_out.shape
            cls_out = cls_out.view(B, 5, 34, H, W)
        else:
            cls_out = torch.sigmoid(cls_out)                     # 用于二分类，很奇怪RaDelft原代码里没有sigmoid
        return cls_out
    

# 梯度反转层（缓解任务冲突）
class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.alpha * grad_output.neg(), None  # 反转梯度


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if (out.max() <= 0.0):
            print('Relu dead!')

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, adapkc=False):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.adapkc = adapkc

        if adapkc:
            self.proj = nn.Sequential(
                # AdaPeakConv2D(in_chans, embed_dim//2, bias=True, refer_band=1, gb_max=3),
                nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
                nn.BatchNorm2d(embed_dim//2),
                nn.GELU(),
                AdaPeakConv2D(embed_dim//2, embed_dim//2, bias=True, refer_band=1, gb_max=3),
                nn.BatchNorm2d(embed_dim//2),
                nn.GELU(),
                AdaPeakConv2D(embed_dim//2, embed_dim, bias=True, refer_band=1, gb_max=3),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 2, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                AdaPeakConv2D(embed_dim, embed_dim, bias=True, refer_band=1, gb_max=3),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
                nn.BatchNorm2d(embed_dim//2),
                nn.GELU(),
                nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
                nn.BatchNorm2d(embed_dim//2),
                nn.GELU(),
                nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)#.permute(0, 2, 3, 1) #(b h w c)
        return x


class BandEst(nn.Module):
    """
    Description:
        *Guard band estimator, BandEst
        *Vertical 1D Conv: (gb_max * 2 + 1, 1);
        *Horizontal 1D Conv: (1, gb_max * 2 + 1);
        input: x in the shape of BxCxHxW
        output: gb_mask in the shape of Bx4xHxW
    """
    def __init__(self, in_ch, gb_max):
        super(BandEst, self).__init__()
        self.in_ch = in_ch
        self.gb_max = gb_max
        self.single_conv_vrt = nn.Conv2d(self.in_ch,  2, kernel_size=(gb_max*2+1, 1), padding=(gb_max, 0), dilation=1)
        self.bn_vrt = nn.BatchNorm2d(2)
        self.single_conv_hrz = nn.Conv2d(self.in_ch,  2, kernel_size=(1, gb_max*2+1), padding=(0, gb_max), dilation=1)
        self.bn_hrz = nn.BatchNorm2d(2)
        # weight initialization
        nn.init.constant_(self.single_conv_vrt.weight, 0)
        nn.init.constant_(self.single_conv_hrz.weight, 0)

    def forward(self, x):
        # (b, 2, h, w)
        gb_map_v = self.bn_vrt(self.single_conv_vrt(x))
        # (b, 2, h, w)
        gb_map_h = self.bn_hrz(self.single_conv_hrz(x))
        # modulate to (0, self.gb_max - 1).
        gb_map_v = torch.sigmoid(gb_map_v) * (self.gb_max - 1)
        gb_map_h = torch.sigmoid(gb_map_h) * (self.gb_max - 1)
        # (b, 4, h, w)
        # in the direction of vt-vb-hl-hr
        return torch.cat((gb_map_v, gb_map_h), 1)

class AdaPeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, gb_max=3):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            signal_type: str in ('range_doppler', 'angle_doppler', 'range_angle').
            gb_max: the upper bound of guard bandwidth in four directions.
        """
        super(AdaPeakConv2D, self).__init__()
        # self.signal_type = signal_type
        self.in_ch = in_ch
        self.out_ch = out_ch
        # the initialized guard bandwidth, also the lower bound of guard bandwidth
        self.init_guard_band = (1, 1)
        # padding order: l-r-t-b
        self.padding = (self.init_guard_band[1] + refer_band,
                        self.init_guard_band[1] + refer_band,
                        self.init_guard_band[0] + refer_band,
                        self.init_guard_band[0] + refer_band)
        self.refer_band = refer_band
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        self.rep_padding = nn.ReplicationPad2d(self.padding)
        # the estimation network of guard bandwidth
        self.gb_estimator = BandEst(self.in_ch, gb_max)
        
        # conv block for peak receptive field (PRF) in x
        self.kernel_size = (self.init_guard_band[0]+self.init_guard_band[1]+self.refer_band*2), 4
        self.stride = self.kernel_size
        self.bias = bias
        self.peak_conv = nn.Conv2d(self.in_ch, self.out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)

    # update p_r with initialized p_r and estimated guard bandwidth offsets
    def get_pr_learned(self, pr, gb_mask):
        b, N, h, w = pr.size()
        gb_x_t = - gb_mask[:, 0, :, :].view(b, 1, h, w)
        gb_x_b = gb_mask[:, 1, :, :].view(b, 1, h, w)
        gb_y_l = - gb_mask[:, 2, :, :].view(b, 1, h, w)
        gb_y_r = gb_mask[:, 3, :, :].view(b, 1, h, w)
        zero = torch.zeros([int(b), int(N/8), int(h), int(w)], device=pr.device)
        move = torch.cat([gb_x_t.repeat(1, int(N/8), 1, 1), zero,
                          gb_x_b.repeat(1, int(N/8), 1, 1), zero,
                          zero, gb_y_r.repeat(1, int(N/8), 1, 1),
                          zero, gb_y_l.repeat(1, int(N/8), 1, 1)], 1)
        pr = pr + move
        return pr

    def forward(self, x):
        b, _, h, w = x.size()
        # kernel_size for pkc
        k_pkc = self.kernel_size
        # prf size
        N = k_pkc[0] * k_pkc[1]
        #dtype = x.data.type()
        dtype = torch.float
        device = x.device

        x_pad = self.rep_padding(x)
        # p_c denotes the center point positions
        p_c = self._get_p_c(b, h, w, N, dtype, device)
        # p_r denotes the initialized reference point positions
        # p_r shape: Bx2NxHxW
        p_r = self._get_p_r(N, p_c, dtype, device)
        # sample x_c (b,c,h,w,N) using p_c
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p_c = p_c.contiguous().permute(0, 2, 3, 1).floor().long()
        x_c = self._sample_x(x_pad, p_c, N)

        # move initialized p_r with estimated gb_mask
        # estimate guard bandwidth offsets.
        # shape pf (b, 4, h, w)
        gb_mask = self.gb_estimator(x)
        # update p_r with offsets and initialized p_r
        # shape of (b, 2N, h, w)
        p = self.get_pr_learned(p_r, gb_mask)

        # put channel dimension at the last
        # shape of (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # let linear interpolation operate on q = [q_lt, q_rb] for each p
        # q in left/top position is the nearest integer number of p
        q_lt = p.detach().floor()
        # q in right/below position = 1 + q in left/top position
        q_rb = q_lt + 1

        # restrict q in the range of [(0, 0), (H-1, W-1)], the H and W is the shape of padded x.
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_lt[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_rb[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()

        # restrict p in the range of [(0, 0), (H-1, W-1)]
        p = torch.cat([torch.clamp(p[..., :N], 0, x_pad.size(2)-1), torch.clamp(p[..., N:], 0, x_pad.size(3)-1)], dim=-1)

        # using q to get linear kernel weights for p
        # shape of (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        # getting x(q) from input x
        # shape of (b, c, h, w, N)
        x_q_lt = self._sample_x(x_pad, q_lt, N)
        x_q_rb = self._sample_x(x_pad, q_rb, N)

        # weighted summation of linear kernel interpolation
        # (b, c, h, w, N)
        x_r = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb

        # getting x_prf for peak convolution
        x_prf = x_c - x_r
        x_prf = self._reshape_x_prf(k_pkc[0], k_pkc[1], x_prf)

        # peak convolution
        out = self.peak_conv(x_prf)
        return out

    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N, dtype, device
                 ):
        # generating pc_grid
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: 1x2NxHxW
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype).to(device)
        # (b,2N,h,w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    # generating peak receptive field grid
    def _gen_prf_grid(self, rb, gb, N, dtype, device):
        # h for row (x); w for col (y)
        h_t = -(rb + gb[0])
        h_d = rb + gb[0]
        w_l = -(rb + gb[1])
        w_r = rb + gb[1]
        # width and height of receptive field
        w_prf = (rb + gb[1]) * 2 + 1
        h_prf = (rb + gb[0]) * 2 + 1

        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1))

        # taking positions clockwise 
        prf_xt = prf_x_idx[0:rb, 0:(w_prf - rb)]
        prf_xr = prf_x_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_xl = prf_x_idx[rb:h_prf, 0:rb]

        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)

        prf_yt = prf_y_idx[0:rb, 0:(w_prf - rb)]
        prf_yr = prf_y_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_yl = prf_y_idx[rb:h_prf, 0:rb]

        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype).to(device)
        return prf

    # getting p_r positions from each p_c
    def _get_p_r(self, N, p_c, dtype, device):
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(self.refer_band, self.init_guard_band, N, dtype, device)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r

    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, h_pad, w_pad)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf
    

class AdaPeak2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, gb_max=3):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            signal_type: str in ('range_doppler', 'angle_doppler', 'range_angle').
            gb_max: the upper bound of guard bandwidth in four directions.
        """
        super(AdaPeak2D, self).__init__()
        # self.signal_type = signal_type
        self.in_ch = in_ch
        self.out_ch = out_ch
        # the initialized guard bandwidth, also the lower bound of guard bandwidth
        self.init_guard_band = (1, 1)
        # padding order: l-r-t-b
        self.padding = (self.init_guard_band[1] + refer_band,
                        self.init_guard_band[1] + refer_band,
                        self.init_guard_band[0] + refer_band,
                        self.init_guard_band[0] + refer_band)
        self.refer_band = refer_band
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        self.rep_padding = nn.ReplicationPad2d(self.padding)
        # the estimation network of guard bandwidth
        self.gb_estimator = BandEst(self.in_ch, gb_max)
        
        # conv block for peak receptive field (PRF) in x
        self.kernel_size = (self.init_guard_band[0]+self.init_guard_band[1]+self.refer_band*2), 4
        self.stride = self.kernel_size
        self.bias = bias

    # update p_r with initialized p_r and estimated guard bandwidth offsets
    def get_pr_learned(self, pr, gb_mask):
        b, N, h, w = pr.size()
        gb_x_t = - gb_mask[:, 0, :, :].view(b, 1, h, w)
        gb_x_b = gb_mask[:, 1, :, :].view(b, 1, h, w)
        gb_y_l = - gb_mask[:, 2, :, :].view(b, 1, h, w)
        gb_y_r = gb_mask[:, 3, :, :].view(b, 1, h, w)
        zero = torch.zeros([int(b), int(N/8), int(h), int(w)], device=pr.device)
        move = torch.cat([gb_x_t.repeat(1, int(N/8), 1, 1), zero,
                          gb_x_b.repeat(1, int(N/8), 1, 1), zero,
                          zero, gb_y_r.repeat(1, int(N/8), 1, 1),
                          zero, gb_y_l.repeat(1, int(N/8), 1, 1)], 1)
        pr = pr + move
        return pr

    def forward(self, x):
        b, _, h, w = x.size()
        # kernel_size for pkc
        k_pkc = self.kernel_size
        # prf size
        N = k_pkc[0] * k_pkc[1]
        #dtype = x.data.type()
        dtype = torch.float
        device = x.device

        x_pad = self.rep_padding(x)
        # p_c denotes the center point positions
        p_c = self._get_p_c(b, h, w, N, dtype, device)
        # p_r denotes the initialized reference point positions
        # p_r shape: Bx2NxHxW
        p_r = self._get_p_r(N, p_c, dtype, device)
        # sample x_c (b,c,h,w,N) using p_c
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p_c = p_c.contiguous().permute(0, 2, 3, 1).floor().long()
        x_c = self._sample_x(x_pad, p_c, N)

        # move initialized p_r with estimated gb_mask
        # estimate guard bandwidth offsets.
        # shape pf (b, 4, h, w)
        gb_mask = self.gb_estimator(x)
        # update p_r with offsets and initialized p_r
        # shape of (b, 2N, h, w)
        p = self.get_pr_learned(p_r, gb_mask)

        # put channel dimension at the last
        # shape of (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # let linear interpolation operate on q = [q_lt, q_rb] for each p
        # q in left/top position is the nearest integer number of p
        q_lt = p.detach().floor()
        # q in right/below position = 1 + q in left/top position
        q_rb = q_lt + 1

        # restrict q in the range of [(0, 0), (H-1, W-1)], the H and W is the shape of padded x.
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_lt[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_rb[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()

        # restrict p in the range of [(0, 0), (H-1, W-1)]
        p = torch.cat([torch.clamp(p[..., :N], 0, x_pad.size(2)-1), torch.clamp(p[..., N:], 0, x_pad.size(3)-1)], dim=-1)

        # using q to get linear kernel weights for p
        # shape of (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        # getting x(q) from input x
        # shape of (b, c, h, w, N)
        x_q_lt = self._sample_x(x_pad, q_lt, N)
        x_q_rb = self._sample_x(x_pad, q_rb, N)

        # weighted summation of linear kernel interpolation
        # (b, c, h, w, N)
        x_r = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb

        # getting x_prf for peak convolution
        x_prf = x_c - x_r
        # peak convolution
        return x_prf

    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N, dtype, device
                 ):
        # generating pc_grid
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: 1x2NxHxW
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype).to(device)
        # (b,2N,h,w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    # generating peak receptive field grid
    def _gen_prf_grid(self, rb, gb, N, dtype, device):
        # h for row (x); w for col (y)
        h_t = -(rb + gb[0])
        h_d = rb + gb[0]
        w_l = -(rb + gb[1])
        w_r = rb + gb[1]
        # width and height of receptive field
        w_prf = (rb + gb[1]) * 2 + 1
        h_prf = (rb + gb[0]) * 2 + 1

        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1))

        # taking positions clockwise 
        prf_xt = prf_x_idx[0:rb, 0:(w_prf - rb)]
        prf_xr = prf_x_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_xl = prf_x_idx[rb:h_prf, 0:rb]

        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)

        prf_yt = prf_y_idx[0:rb, 0:(w_prf - rb)]
        prf_yr = prf_y_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_yl = prf_y_idx[rb:h_prf, 0:rb]

        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype).to(device)
        return prf

    # getting p_r positions from each p_c
    def _get_p_r(self, N, p_c, dtype, device):
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(self.refer_band, self.init_guard_band, N, dtype, device)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r

    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, h_pad, w_pad)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf
    

class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer,out_layer,kernel_size=(1,12),dilation=(1,16),use_bn = False, adaptive_pkc=False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn
        self.adaptive_pkc = adaptive_pkc

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size, 
                              stride=(1, 1), padding=0,dilation=dilation, bias= (not use_bn) )
     
        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)


    def forward(self,x):
        width = x.shape[-1]
        x = torch.cat([x[...,-self.padding:],x,x[...,:self.padding]],axis=3)
        x = self.conv(x)
        x = x[...,int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
  
        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x, aux=False):
        '''
        x: B H W C
        '''
        if not aux:
            x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        if not aux:
            x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
    

class DoubleAdaPKC2D(nn.Module):
    """ (2D AdaPKC => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, bias, rb, gb_max=3):
        super(DoubleAdaPKC2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.gb_max = gb_max
        self.pk_conv1 = AdaPeakConv2D(in_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      gb_max=self.gb_max)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = AdaPeakConv2D(out_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      gb_max=self.gb_max)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.pk_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pk_conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    

class BasicLayer(nn.Module):  

    def __init__(self, layer_id, num_layer, embed_dim, out_dim, depth, num_heads, chunkwise_recurrent=False, drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample: PatchMerging=None, use_checkpoint=False, adaptive_pkc=False, peak_attn=False):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        self.adaptive_pkc = adaptive_pkc
        self.peak_attn = peak_attn
        print("peakattn: ", peak_attn, "decompose: ", chunkwise_recurrent, "adapkc: ", adaptive_pkc)
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
            
        # build blocks
        self.blocks = nn.ModuleList([
            RwkvBlock(flag, layer_id=layer_id ,num_layer=num_layer , dim=embed_dim, num_heads=num_heads,  
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, adaptive_pkc=adaptive_pkc,
                     peak_attn=peak_attn)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, c = x.size()
        # rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        # if self.adaptive_pkc:
        #     x_pkc = self.pk_conv1(x.permute(0,3,1,2))
        #     x_pkc = self.bn1(x_pkc)
        #     x_pkc = self.act1(x_pkc)
        #     x = x + x_pkc.permute(0,2,3,1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x)
            else:
                x = blk(x)
            
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x

class Quadratic(torch.nn.Module):
    def __init__(self, size_in, size_out):
        """
        In the constructor we instantiate three parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.linear = nn.Linear(size_in, size_out) 
        # self.linear = nn.Conv2d(size_in, size_out, 3, 1, 1)
        self.quadratic = nn.Linear(size_in, size_out,bias=False) 
        # self.quadratic =nn.Conv2d(size_in, size_out, 3, 1, 1, bias=False)

        self.bias = self.linear.bias
        self.weight_linear = self.linear.weight
        self.weight_quadratic = self.quadratic.weight

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        B, C, H, W = x.shape
        # x = x.reshape(B, C, H*W).permute(0,2,1)
        # x = x.flatten(2)  # (B, C, H*W)
        x = x.view(B, C, H * W)
        # x = x.transpose(1, 2).contiguous()  # (B, H*W, C)
        x = x.permute(0,2,1).contiguous()
        x = self.linear(x) + self.quadratic(x**2)
        # x = x.permute(0,2,1).reshape(B,1,H,W)
        # x = x.transpose(1, 2).contiguous().view(B, 1, H, W)
        x = x.permute(0,2,1).contiguous()
        x = x.view(B, 1, H, W)
        return x

class Exponential(nn.Module):
    """
    Simple exponential activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)

class VIBBottleneck(nn.Module):
    r"""NVIB from https://github.com/idiap/nvib.git"""
    def __init__(self, size_in, size_out, prior_mu=None, prior_var=None, prior_log_alpha=None, prior_log_alpha_stdev=None,
        delta=1, kappa=1, nheads=1, alpha_tau=None, stdev_tau=None, mu_tau=None,):
        super().__init__()
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Priors
        self.prior_mu = (prior_mu if prior_mu is not None else torch.zeros(size_in)).to(
            self.device
        )  # [H]
        self.prior_var = (prior_var if prior_var is not None else torch.ones(size_in)).to(
            self.device
        )  # [H]
        self.prior_log_alpha = (
            prior_log_alpha if prior_log_alpha is not None else torch.zeros(1)
        ).to(
            self.device
        )  # [1]
        self.prior_log_alpha_stdev = (
            prior_log_alpha_stdev if prior_log_alpha_stdev is not None else torch.ones(1)
        ).to(
            self.device
        )  # [1]

        self.prior_var = self.prior_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.prior_mu = self.prior_mu.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        self.delta = float(delta)  # Conditional prior delta
        self.kappa = int(kappa)  # Number of samples

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        # self.d = int(size_in / nheads)  # dimension of the head
        self.alpha_activation = Exponential()  # projection for alphas
        # self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.mu_proj = nn.Conv2d(size_in, size_out, 3, 1, 1)
        # self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.logvar_proj = nn.Conv2d(size_in, size_out, 3, 1, 1)
        self.alpha_proj = Quadratic(size_in, 1)  # Project to model size
        # self.alpha_bn = nn.BatchNorm2d(size_in)

        # Initialisation parameters - 0 is the prior 1 is the posterior
        self.alpha_tau = alpha_tau if alpha_tau is not None else 1
        self.stdev_tau = stdev_tau if stdev_tau is not None else 1
        self.mu_tau = mu_tau if mu_tau is not None else 1
        self.apply(self.init_weights)    #从哪参考的了？？？？
        
    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # VIB的mu和logvar层初始化
            if hasattr(module, 'weight'):
                # nn.init.xavier_uniform_(module.weight, gain=0.02)  # 小增益初始化，0.1还是0.01？？？？
                nn.init.xavier_uniform_(module.weight, gain=1) 
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # if isinstance(module, nn.BatchNorm2d):                    # 后加的BN的初始化
        #     nn.init.normal_(module.weight, 1.0, 0.02)
        #     nn.init.constant_(module.bias, 0.0)

    def reparameterize_gaussian(self, mu, logvar):
        """
        Reparameterise for gaussian
        Train = sample
        Test = mean

        :param mu: means [Nl,B,H]
        :param logvar: logged variances [Nl,B,H]
        :return: z: sample from a gaussian distribution or mean
        """

        if self.training:
            std = torch.exp(0.5 * logvar)  # [Nl,B,H]
            eps = torch.randn_like(std)  # [Nl,B,H]
            z = eps.mul(std).add_(mu)  # [Nl,B,H]
        else:
            z = mu  # [Nl,B,H]
        return z  # [Nl,B,H]

    def reparameterize_dirichlet(self, alpha):
        """
        Takes in alpha parameters and returns pi from a dirichlet distribution.

        :param alpha: [Nl,B,1]
        :param mask: Mask for the latent space [B,Nl]
        :return: pi [Nl,B,1]
        """

        if self.training:
            if torch.isinf(alpha).any():
                print("has_inf in alpha!")
            elif torch.isnan(alpha).any():
                print("has_nan in alpha!")
            # Implicit gradients for Gamma (batch_shape [Nl, B]) each individual gamma
            gamma_dist = torch.distributions.Gamma(alpha, torch.ones_like(alpha))
            gammas = gamma_dist.rsample()

        # Testing the alphas don't have noise
        else:
            thresh = nn.Threshold(0.1, 0)
            gammas = thresh(alpha)

        # mask and normalise (make sure its non-zero)
        # gammas.masked_fill_(mask, 0)
        normalising_sum = torch.sum(gammas, 0).unsqueeze(0) + torch.finfo(gammas.dtype).tiny
        pi = torch.div(gammas, normalising_sum)

        return pi
    
    def kl_gaussian(self, mu, logvar, alpha):
        """
        KL Loss for the Gaussian component with expected K
        :param mu: mean [Nl,B,H]
        :param logvar: logged variance [Nl,B,H]
        :param alpha: psuedo count weight [Nl,B,1]
        :param memory_key_padding_mask: boolean mask [B,Nl]
        :return: KL [B]
        """

        alpha0_q = torch.sum(alpha.transpose(2, 0), -1)  # [1,B]
        expected_pi = alpha.squeeze(-1) / alpha0_q  # [Nl,B]

        # KL between univariate Gaussians
        var_ratio = logvar.exp() / self.prior_var
        t1 = (mu - self.prior_mu) ** 2 / self.prior_var
        kl = var_ratio + t1 - 1 - var_ratio.log()
        # kl = kl.masked_fill(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        if torch.isnan(kl).any() or torch.isinf(kl).any():
            print("var_ratio: ", var_ratio)
            print("var_ratio_log: ", var_ratio.log())
            # raise ValueError("kl 包含 NaN 或 inf!")

        # Mean over embedding dimension
        B, C, H, W = kl.shape
        kl = torch.mean(kl, 1).reshape(B, H*W).permute(1,0)  # [Nl,B]

        # Scale and sum over sentence length dimension
        # kl = 0.5 * k0 * torch.sum(kl * expected_pi, 0) / n 
        kl = 0.5 * torch.sum(kl * expected_pi, 0)
        if torch.isnan(kl).any() or torch.isinf(kl).any():
            print("kl: ", kl)
            # raise ValueError("kl 包含 NaN 或 inf!")

        return kl

    def kl_dirichlet(self, alpha, memory_key_padding_mask):
        """
        The regularisation for the dirichlet component with expected K

        :param alpha: k dimensional psuedo counts [Nl,B,1]
        :param memory_key_padding_mask: boolean mask [B,Nl]
        :return: Kl [B]

        Nota Bene: digamma and lgamma cannot be zero
        """

        # # Total number of vectors sampled
        k0 = torch.sum(~memory_key_padding_mask.transpose(1, 0), 0)  # [B]
        # # Input length
        n = k0 / self.kappa  # [B]
        # # Conditional prior lower bound. Sentence length without prior
        lowerBound = self.delta * (n - 1)

        # Sum the alphas
        alpha0_q = torch.sum(alpha, 0).squeeze(-1).to(torch.float64)  # [B]
        alpha0_p = (torch.ones_like(alpha0_q) * (torch.exp(self.prior_log_alpha) + lowerBound)).to(
            torch.float64
        )  # [B]

        kl = (
            torch.lgamma(alpha0_q)
            - torch.lgamma(alpha0_p)
            + (alpha0_q - alpha0_p) * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
            + k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
        ) / n                     #假设了均匀分布？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        if torch.isnan(kl).any() or torch.isinf(kl).any():
            print("lg_q: ", torch.lgamma(alpha0_q))
            print("lg_p: ", torch.lgamma(alpha0_p))
            print("dg_q: ", torch.digamma(alpha0_q))
            print("dg_p: ", torch.digamma(alpha0_p))

        return kl.to(torch.float32)
    
    def forward(self, encoder_output, mask=None, alpha_skip=None, kl_loss=True):
        # B, C, H, W = encoder_output.shape
        # Project to mean, log variance and log alpha
        B, C, H, W = encoder_output.shape
        mu = self.mu_proj(encoder_output)
        logvar = self.logvar_proj(encoder_output)

        # Alpha skip connection in log space
        if alpha_skip is not None:
            log_alpha = self.alpha_proj(encoder_output) + torch.log(alpha_skip[1:, :, :])
            alpha = self.alpha_activation(log_alpha)
        else:
            log_alpha = self.alpha_proj(encoder_output)
            alpha = self.alpha_activation(log_alpha)
        # alpha = torch.clamp(alpha, min=0, max=torch.finfo(alpha.dtype).max - 1000)
        alpha = alpha.view(B, 1, H*W).permute(2,0,1).contiguous()
        # Reparameterise
        z = self.reparameterize_gaussian(mu, logvar)
        pi = self.reparameterize_dirichlet(alpha)

        if kl_loss:
            kl_g = self.kl_gaussian(mu, logvar, alpha)
            kl_d = self.kl_dirichlet(alpha, mask)
            return z, kl_g, kl_d, alpha, pi
        else:
            return z, None, None, alpha, pi

        

    def sample(self, number_samples, memory_key_padding_mask, device, *args, **kwargs):
        r"Sampling is done when the model is in evaluation mode (no dropout)."

        # Sample from a gaussian
        memory_key_padding_mask = memory_key_padding_mask.repeat(1, self.kappa)
        max_length = memory_key_padding_mask.size(-1)
        eps = torch.randn(
            size=(max_length, number_samples, self.size_out), device=device
        )  # [Ns,B,H]
        z = self.prior_mu + (self.prior_var**0.5) * eps
        z.masked_fill_(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        logvar = torch.ones_like(z) * -200  # When exponentiated it will be 0

        # Sample from Dir((alpha1 + K0 * delta)/K0, ... )
        # When delta = 0 (Dirichlet process) Dir((alpha0/K0, ... ,alpha0/K0)
        # When delta = 1 (Full conditional prior) Dir((alpha0, ... ,alpha0)
        K0 = torch.sum(~memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        alphas = (
            torch.ones(size=(max_length, number_samples, 1), device=device)
            * (self.prior_alpha + (self.delta * (K0 - 1)))
            / K0
        )
        alphas.masked_fill_(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        pi = self.reparameterize_dirichlet(alphas, memory_key_padding_mask.T.unsqueeze(-1))

        # This is how the decoder gets the parameters
        z_tuple = (z, pi, z, logvar)

        return z_tuple, memory_key_padding_mask


class RMT(nn.Module):

    def __init__(self, in_chans=3, out_indices=(0, 1, 2, 3), peak_attn=[True, False, False, False], 
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 4, 2], num_heads=[4, 4, 8, 16], adapkc=[True, True, True, True], 
                 init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, True, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True,
                 ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.patch_dim = 32
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.norm_eval = norm_eval

        norm_dims = [64, 128, 256, 256]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.patchembed = PatchEmbed(in_chans=in_chans, embed_dim=self.patch_dim, adapkc=False)
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                layer_id = i_layer,
                num_layer = self.num_layers,
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                adaptive_pkc=adapkc[i_layer],
                peak_attn=peak_attn[i_layer]
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.extra_norms.append(nn.LayerNorm(norm_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        # z, ex_recon, dictionary = self.sps_cd(x)
        x = self.patchembed(x).permute(0,2,3,1)
        # mask = torch.abs(z) > 1e-5
        # print(mask.sum())
        # latent_loss = torch.sum((ex_recon - x).pow(2), dim=1).mean()

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x) #x_out->[b,64,64,32]; x->[b,32,32,64]
            if i in self.out_indices:
                x = self.extra_norms[i](x)#x_out->[b,64,64,32]
                out = x.permute(0, 3, 1, 2).contiguous()
                outs.append(out) #[b,32,64,64],[b,64,32,32],[b,128,16,16],[b,256,8,8]
        
        return outs

    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class AuxEncoder(nn.Module):
    def __init__(self, in_channels): 
        super().__init__()
        
        self.patchembed = PatchEmbed(in_chans=128, embed_dim=32, adapkc=False)
        self.stages = nn.ModuleList([
            ResidualBlock(ch, ch, PatchMerging) for ch in in_channels[:3]
        ])
        self.stages.append(ResidualBlock(in_channels[3], in_channels[3], None))

        # self.bn = nn.ModuleList([nn.BatchNorm2d(num_features=ch) for ch in [64, 128, 256, 256]])  #改了这里，多余
        # self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels[3])
        
    def forward(self, x, ret):
        x = self.patchembed(x)
        layer_outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i==3:
                x = self.bn(x)
            if ret[i]:
                layer_outputs.append(x)
            else:
                layer_outputs.append(nn.Identity())
        
        return layer_outputs


class FusionEnc(nn.Module):
    def __init__(self, in_channels, ret):
        super().__init__()

        self.ret = ret
      
        self.fusion =nn.ModuleList()
        self.fuse_post = nn.ModuleList()
        for i, ch in enumerate(in_channels):
            if ret[i]:
                self.fusion.append(nn.Sequential(
                    nn.Conv2d(ch, ch//4, 3, 1, 1),
                    nn.BatchNorm2d(ch//4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch//4, ch//2, 3, 1, 1),
                    nn.BatchNorm2d(ch//2),
                    # nn.ReLU(inplace=True)   #最后一层relu要不要加？？？？
                ))
                self.fuse_post.append(nn.BatchNorm2d(num_features=ch//2))
            else:
                self.fusion.append(nn.Identity())
                self.fuse_post.append(nn.Identity())

        self.residule = nn.Identity()
        # self.fuse_post = nn.ModuleList([nn.BatchNorm2d(num_features=nf) for nf in [64, 128, 256, 256]])

    def forward(self, feat_power, feat_ele, vib=None, kl_factor=0.0, kl_loss=True):
        feat_fused = []
        klg_loss = []
        kld_loss = []
        alpha_list = []
        
        for i in range(len(feat_power)):
            if self.ret[i]:
                concat_feat = torch.cat([feat_power[i], feat_ele[i]], dim=1)
                concat_feat = self.fusion[i](concat_feat)                
                concat_feat = concat_feat + self.residule(feat_power[i])
                concat_feat = self.fuse_post[i](concat_feat)
                if vib is not None:
                    mask = torch.zeros(feat_power[i].shape[0], feat_power[i].shape[2]*feat_power[i].shape[3], dtype=bool).to('cuda')
                    concat_feat, kl_g, kl_d, alpha, pi = vib(concat_feat, mask=mask, kl_loss=kl_loss)
                    if kl_factor > 0.0:
                        B, C, H, W = concat_feat.shape
                        pi = pi.permute(1,2,0).reshape(B, -1, H, W)
                        concat_feat += pi * kl_factor              #KL_factor不等于0时开始加入

                    if kl_loss:
                        klg_loss.append(kl_g)
                        kld_loss.append(kl_d)
                    else:
                        klg_loss = None
                        kld_loss = None
                    alpha_list.append(alpha)
            else:
                concat_feat = self.residule(feat_power[i])
            feat_fused.append(concat_feat)

        if vib is not None:
            return feat_fused, klg_loss, kld_loss, alpha_list
        else:
            return feat_fused


class SparseRadarRWKV(nn.Module):
    def __init__(self, regression_layer=2, out_dim=224, patch_norm=True, init=True, detection_head=True, 
                 segmentation_head=True, num_atoms=32, num_dims=32, num_iters=16, device='cpu', multiclass=False):
        
        super().__init__()

        self.patch_norm =patch_norm
        self.out_dim = out_dim
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        input_channels = 128   

        self.power_bn = nn.BatchNorm2d(input_channels)
        self.ele_bn = nn.BatchNorm2d(input_channels)                       #改了这里
        
        self.backbone = RMT(in_chans=input_channels, out_indices=(0, 1, 2, 3), peak_attn=[True, False, False, False],
                             embed_dims=[32, 64, 128, 256], depths=[2, 2, 4, 2], num_heads=[4, 4, 8, 16], adapkc=[False, False, False, False],
                             init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                             patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, True, False], projection=1024,
                             layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True)

        self.aux_backbone = AuxEncoder(in_channels=[32, 64, 128, 256])

        # self.chan_expend = nn.ModuleList([nn.Conv2d(channels, channels//2, kernel_size=1) for channels in [128, 256, 512, 512]])
        self.ret_layer = [False, False, False, True]
        self.fuse_backbone = FusionEnc(in_channels=[128, 256, 512, 512], ret=self.ret_layer)      #改了这里
        
        # self.fuse_act = nn.ReLU(inplace=True)
        # self.vib_bottleneck = nn.ModuleList([VIBBottleneck(in_channels=input_channels, latent_dim=input_channels, beta=0.01) 
        #                                      for input_channels in [64, 128, 256, 256]])
        self.vib_bottleneck = VIBBottleneck(size_in=256, size_out=256, kappa=1.0, delta=0.25)
        # self.vib_bottleneck = None
        self.RAE_decoder = RAEDecoder()

        self.detection_header = DetHeader(multiclass=multiclass)
        # self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

        if init:
            self._init_weights(self.aux_backbone)
            self._init_weights(self.fuse_backbone)
            self._init_weights(self.RAE_decoder)
            self._init_weights(self.detection_header)


    def _init_weights(self, module=None, init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                # torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')    #改了这里
                if m.bias is not None:                                                    #改了这里
                    nn.init.constant_(m.bias, 0)                                          #改了这里
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.constant_(m.bias, 0.0)
        
        if module is None:
            module = self
        # apply 会自动递归遍历所有子模块
        module.apply(init_func)


    def forward(self, x, current_alpha=None, kl_factor=0.0, kl_loss=True):
        # input: [B, 128, 512, 256] <==> [B, D, R, A]
        # x = self.init_bn(x)
        # visualize_3d(x[:,0,:,:,:])
        x_power = self.power_bn(x[:,0,:,:,:])
        x_ele = self.ele_bn(x[:,1,:,:,:])
        
        feats_power = self.backbone(x_power)
        feats_ele = self.aux_backbone(x_ele, self.ret_layer)
       
        feats_fused, klg_loss, kld_loss, alpha = self.fuse_backbone(feats_power, feats_ele, 
                                                                    vib=self.vib_bottleneck, 
                                                                    kl_factor=kl_factor,
                                                                    kl_loss=kl_loss)    #改了这里
        
        RA = self.RAE_decoder(feats_fused)

        out = self.detection_header(RA)

        return out, klg_loss, kld_loss, alpha

      
class RAEDecoder(nn.Module):
    def __init__(self, ):
        super(RAEDecoder, self).__init__()
        
        ele_out = 128

        self.conv_block1 = BasicBlock(256, ele_out)
        self.conv_block2 = BasicBlock(256, ele_out)
        self.conv_block3 = BasicBlock(256, ele_out)

        self.deconv11 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv12 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv13 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv14 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv15 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv16 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.deconv17 = nn.ConvTranspose2d(ele_out, ele_out, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.L5 = nn.Conv2d(256,ele_out,kernel_size=1,stride=1,padding=0)
        self.L4 = nn.Conv2d(256,ele_out,kernel_size=1,stride=1,padding=0)
        self.L3 = nn.Conv2d(128, ele_out, kernel_size=1, stride=1,padding=0)
        self.L2 = nn.Conv2d(64, ele_out, kernel_size=1, stride=1,padding=0)
        
        
    def forward(self,features):
        
        T5 = self.L5(features[3])
        T4 = self.L4(features[2])
        T3 = self.L3(features[1])
        T2 = self.L2(features[0])

        S3 = torch.cat((self.deconv11(T5), self.deconv12(T4)), axis=1)
        S2 = torch.cat((self.deconv13(self.conv_block1(S3)), self.deconv14(T3)), axis=1)
        S1 = torch.cat((self.deconv15(self.conv_block2(S2)), self.deconv16(T2)), axis=1)
        out = self.deconv17(self.conv_block3(S1))

        return out


class HierarchicalMIBCriterion(nn.Module):
    """
    分层MIB损失函数：组合所有层的MIB损失
    """
    def __init__(self, layer_weights=None, beta_params=None):
        super().__init__()
        
        # 各层损失权重（默认为指数衰减）
        self.layer_weights = layer_weights or [0.125, 0.25, 0.5, 1.0]
        
        # 各层β参数（可调整）
        self.beta_params = beta_params or [0.01, 0.05, 0.1, 0.2]
        
        # 任务损失（如果有监督信号）
        self.task_criterion = nn.CrossEntropyLoss()
    
    def forward(self, model_outputs, labels=None):
        """
        model_outputs: 包含'mib_losses'等的字典
        labels: 监督标签（可选）
        """
        # 1. 分层MIB损失
        mib_losses = model_outputs['mib_losses']
        total_mib_loss = 0
        
        for i, layer_loss in enumerate(mib_losses):
            weight = self.layer_weights[i] if i < len(self.layer_weights) else 1.0
            total_mib_loss += weight * layer_loss
        
        # 2. 任务损失（如果有监督标签）
        task_loss = 0
        if labels is not None:
            final_rep = model_outputs['final_representation']
            
            # 假设有任务头（分类头）
            if not hasattr(self, 'task_head'):
                # 动态创建任务头
                input_dim = final_rep.shape[1]
                self.task_head = nn.Linear(input_dim, labels.shape[1]).to(final_rep.device)
            
            task_pred = self.task_head(final_rep)
            task_loss = self.task_criterion(task_pred, labels)
        
        # 3. 可选：特征多样性损失（防止崩溃）
        diversity_loss = self.feature_diversity_loss(model_outputs['fused_features'])
        
        # 4. 总损失
        total_loss = (
            total_mib_loss + 
            1.0 * task_loss +  # 任务损失权重
            0.01 * diversity_loss  # 多样性正则化
        )
        
        return {
            'total_loss': total_loss,
            'mib_loss': total_mib_loss,
            'task_loss': task_loss,
            'diversity_loss': diversity_loss
        }
    
    def feature_diversity_loss(self, fused_features):
        """
        鼓励不同层特征具有多样性
        防止所有层学习相同的特征
        """
        loss = 0
        
        # 计算不同层特征之间的相关性
        for i in range(len(fused_features)):
            for j in range(i+1, len(fused_features)):
                feat_i = fused_features[i]
                feat_j = fused_features[j]
                
                # 展平特征图
                if feat_i.dim() == 4:
                    feat_i = feat_i.view(feat_i.shape[0], feat_i.shape[1], -1)
                    feat_j = feat_j.view(feat_j.shape[0], feat_j.shape[1], -1)
                
                # 计算特征相关性（希望相关性低）
                correlation = self.compute_correlation(feat_i, feat_j)
                loss += torch.mean(torch.abs(correlation))
        
        return loss / (len(fused_features) * (len(fused_features) - 1) / 2)
    
    def compute_correlation(self, x, y):
        """计算两个特征集之间的相关系数"""
        # 展平批次和空间维度
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        
        # 计算相关系数
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
        
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt(
            (x_centered.pow(2).sum(dim=1)) * 
            (y_centered.pow(2).sum(dim=1))
        ) + 1e-8
        
        correlation = numerator / denominator
        
        return correlation

        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, downsample: PatchMerging=None):
        super(ResidualBlock, self).__init__()
        
        self.downsample = downsample
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hid_channels//4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hid_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_channels//4,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        if downsample:
            self.downsample = downsample(dim=in_channels, out_dim=in_channels*2)
    
    def forward(self, x, res=True):
        if res:
            x = x + self._block(x)   #改一下，最后一层的逻辑有问题
        else:
            x = self._block(x)
        if self.downsample:
            x = self.downsample(x, aux=True)
        return x
    

def visualize_3d(data):
    data = data.cpu()
    _, C, H, W = data.shape
    # 显示三个正交切片
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY平面切片（Z=25）
    axes[0].imshow(data[0, :, :, 25], cmap='viridis')
    axes[0].set_title('XY Slice (Z=25)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # XZ平面切片（Y=25）
    axes[1].imshow(data[0, :, 25, :], cmap='viridis')
    axes[1].set_title('XZ Slice (Y=25)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')

    # YZ平面切片（X=25）
    axes[2].imshow(data[0, 25, :, :], cmap='viridis')
    axes[2].set_title('YZ Slice (X=25)')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')

    plt.colorbar(axes[2].imshow(data[0, 25, :, :], cmap='viridis'), ax=axes, fraction=0.046)
    plt.tight_layout()
    plt.savefig("input.png")
