import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
# from model.my_gelu import GELU

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


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
        return x / torch.sqrt(sigma+1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        'define my gelu'
        # self.gelu = GELU()

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        x = self.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.divide = 4
        dim = int(dim / self.divide)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        _, channel, out_shape, _ = x.shape
        if out_shape > 2:
            kernel_size = 2
            encoder = Encoder(channel, int(channel / self.divide), kernel_size).cuda()
        else:
            kernel_size = 1
            encoder = Encoder(channel, int(channel / self.divide), kernel_size).cuda()
        x = encoder(x)
        y = encoder(y)

        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_s = (q.transpose(-2, -1) @ k) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (v @ attn_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_s = self.project_out(out_s)

        attn_c = (q @ k.transpose(-2, -1)) * self.temperature
        attn_c = attn_c.softmax(dim=-1)
        out_c = (attn_c @ v)
        out_c = rearrange(out_c, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_c = self.project_out(out_c)

        decoder = Decoder(c, self.divide * c, kernel_size).cuda()
        out = decoder(out_s + out_c)
        # if out_shape != out.shape[-1]:
        #     out = F.interpolate(out, (out_shape, out_shape), mode="nearest")
        return out


def Encoder(in_channels, out_channels, kernel_size):
    middle_channel = out_channels // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def Decoder(in_channels, out_channels, kernel_size):
    middle_channel = out_channels // 2
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.ConvTranspose2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.ConvTranspose2d(middle_channel, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):

        super(TransformerBlock, self).__init__()

        # self.conv1 = nn.Conv2d(dim_2, dim, (1,1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S): # input_R 原模型特征 input_S 语义分割特征
        #
        # input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        #
        # input_S = self.conv1(input_S)

        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        # input_R = input_R + self.ffn(self.norm2(input_R))

        return input_R

def Confuse(s_features, t_features):
    features = []
    for input_s, input_t in zip(s_features, t_features):
        _, c, _, _ = input_s.shape
        adj = TransformerBlock(dim=c).cuda()
        x = adj(input_s, input_t)
        features.append(x)
    return features

# def Confuse(model, teacher = ''):
#     fuse = []
#     if 'x4' in model:
#         fuse.append(TransformerBlock(256).cuda())
#         fuse.append(TransformerBlock(256).cuda())
#         fuse.append(TransformerBlock(128).cuda())
#         fuse.append(TransformerBlock(64).cuda())
#     elif 'resnet' in model:
#         fuse.append(TransformerBlock(64).cuda())
#         fuse.append(TransformerBlock(256).cuda())
#         fuse.append(TransformerBlock(128).cuda())
#         fuse.append(TransformerBlock(64).cuda())
#         in_channels = [16, 32, 64, 64]
#         out_channels = [16, 32, 64, 64]
#         shapes = [1, 8, 16, 32, 32]
#         # shapes = [1, 56, 112, 224, 224]
#     elif 'vgg' in model:
#         in_channels = [128, 256, 512, 512, 512]
#         shapes = [1, 4, 4, 8, 16]
#         if 'ResNet50' in teacher:
#             out_channels = [256, 512, 1024, 2048, 2048]
#             out_shapes = [1, 4, 8, 16, 32]
#         else:
#             out_channels = [128, 256, 512, 512, 512]
#     elif 'Mobile' in model:
#         if 'ResNet50' in teacher:
#             # fuse.append(TransformerBlock(2048).cuda())
#             # fuse.append(TransformerBlock(2048).cuda())
#             # fuse.append(TransformerBlock(1024).cuda())
#             # fuse.append(TransformerBlock(512).cuda())
#             # fuse.append(TransformerBlock(256).cuda())
#             fuse.append(TransformerBlock(256).cuda())
#             fuse.append(TransformerBlock(512).cuda())
#             fuse.append(TransformerBlock(1024).cuda())
#             fuse.append(TransformerBlock(2048).cuda())
#             fuse.append(TransformerBlock(2048).cuda())
#         else:
#             out_channels = [128, 256, 512, 512, 512]
#             out_shapes = [1, 4, 4, 8, 16]
#     elif 'ShuffleV1' in model:
#         in_channels = [240, 480, 960, 960]
#         shapes = [1, 4, 8, 16]
#         if 'wrn' in teacher:
#             out_channels = [32, 64, 128, 128]
#             out_shapes = [1, 8, 16, 32]
#         else:
#             out_channels = [64, 128, 256, 256]
#             out_shapes = [1, 8, 16, 32]
#     elif 'ShuffleV2' in model:
#         in_channels = [116, 232, 464, 1024]
#         shapes = [1, 4, 8, 16]
#         out_channels = [64, 128, 256, 256]
#         out_shapes = [1, 8, 16, 32]
#     elif 'wrn' in model:
#         r = int(model[-1:])
#         in_channels = [16 * r, 32 * r, 64 * r, 64 * r]
#         out_channels = [32, 64, 128, 128]
#         shapes = [1, 8, 16, 32]
#     else:
#         assert False
#
#     return fuse