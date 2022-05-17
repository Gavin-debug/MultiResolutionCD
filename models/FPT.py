import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class FPT(nn.Module):
    def __init__(self):
        super(FPT,self).__init__()

        self.conv_x1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, stride=1)
        self.conv_x2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, stride=1)
        self.conv_x2_128 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, stride=2)
        self.conv_x2_64 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, stride=2)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3,3), padding=1, stride=1)
        self.conv_a = nn.Conv2d(32, 4, kernel_size=1,padding=0, bias=False)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.transformer = Transformer(dim=32, depth=1, heads=8,
                                       dim_head=64,mlp_dim=64, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=32, depth=1,
                            heads=8, dim_head=64, mlp_dim=64, dropout=0,
                                                      softmax=True)

    def tokenizer(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, 4, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def transformer_decode(self, x, token):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, token)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):
        x1_64 = self.conv_1(x1)
        x1_128 = self.up_2(x1_64)
        x1_256 = self.up_2(x1_128)
        token_x1_64 = self.tokenizer(x1_64)
        token_x1_128 = self.tokenizer(x1_128)
        token_x1_256 = self.tokenizer(x1_256)
        token_x1 = torch.cat((token_x1_64, token_x1_128, token_x1_256), dim=1)
        token_x1 = self.transformer(token_x1)
        token_x1_64n, token_x1_128n, token_x1_256n = token_x1.chunk(3, dim=1)
        x1_64n = self.transformer_decode(x1_64, token_x1_64n)
        x1_128n = self.transformer_decode(x1_128, token_x1_128n)
        x1_256n = self.transformer_decode(x1_256, token_x1_256n)

        x2_256 = self.conv_x2(x2)
        x2_128 = self.conv_x2_128(x2_256)
        x2_64 = self.conv_x2_64(x2_128)
        token_x2_64 = self.tokenizer(x2_64)
        token_x2_128 = self.tokenizer(x2_128)
        token_x2_256 = self.tokenizer(x2_256)
        token_x2 = torch.cat((token_x2_64, token_x2_128, token_x2_256), dim=1)
        token_x2 = self.transformer(token_x2)
        token_x2_64n, token_x2_128n, token_x2_256n = token_x2.chunk(3, dim=1)
        x2_64n = self.transformer_decode(x2_64, token_x2_64n)
        x2_128n = self.transformer_decode(x2_128, token_x2_128n)
        x2_256n = self.transformer_decode(x2_256, token_x2_256n)

        dm1 = torch.abs(x1_64n - x2_64n)
        dm2 = torch.abs(x1_128n - x2_128n)
        dm3 = torch.abs(x1_256n - x2_256n)

        dm = self.up_4(dm1) + self.up_2(dm2) + dm3
        pred = self.classifier(dm)
        return pred
