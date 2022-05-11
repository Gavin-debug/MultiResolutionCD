import torch.nn as nn
from torch.optim import lr_scheduler
from einops import rearrange
import torch
import numbers
import torch.nn.functional as F


def get_scheduler(optimizer, lr_policy, max_epoches):
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(max_epoches + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif lr_policy == 'step':
        step_size = max_epoches//3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return  scheduler

class Upsample_x2(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_x2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class Upsample_x4(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_x4, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 16, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(4))
    def forward(self, x):
        return self.body(x)

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Multi_conv(nn.Module):
    def __init__(self):
        super(Multi_conv,self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=12, stride=4, padding=4)
        self.embed = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.transformer = Transformer(dim=32, depth=1, heads=8,
        #                                dim_head=64,
        #                                mlp_dim=64, dropout=0)
        # self.transformer_decoder = TransformerDecoder(dim=32, depth=1,
        #                     heads=8, dim_head=64, mlp_dim=64, dropout=0,
        #                                               softmax=True)
        self.transformer = TransformerBlock(dim=32, num_heads=8, ffn_expansion_factor=2.66,bias=False,
                                            LayerNorm_type='WithBias')
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.upsample_x2 = Upsample_x2(32)
        self.upsample_x4 = Upsample_x4(32)
        self.classifier = TwoLayerConv2d(in_channels=96, out_channels=2)


    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.embed(x1)
        x2 = self.conv2(x2)
        x2 = self.embed(x2)  ## n, 32, 64, 64
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)
        x_out_1 = self.upsample_x4(torch.abs(x1 - x2))

        x1 = self.upsample_x2(x1)
        x2 = self.upsample_x2(x2)
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)
        x_out_2 = self.upsample_x2(torch.abs(x1 - x2))

        x1 = self.upsample_x2(x1)
        x2 = self.upsample_x2(x2)
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)
        x_out_3 = torch.abs(x1 - x2)

        x = torch.cat([x_out_1, x_out_2, x_out_3], dim=1)
        x = self.relu(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x