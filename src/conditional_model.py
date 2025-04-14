import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from monai.networks.nets.attentionunet import ConvBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Position Embedding for an amplitude difference.
    Adapted from https://huggingface.co/blog/annotated-diffusion to multi-dim case
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): dimension to embed to.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock_with_time(nn.Module):
    """
        Monai ConvBlock with additional time embedding MLP if required.
        Adapted from Monai code and https://huggingface.co/blog/annotated-diffusion.
        Arguments are similar to Monai class, except for time_emb_dim if given.
    """
    def __init__(self, spatial_dims,in_channels,out_channels,
                time_emb_dim=None, kernel_size=3, strides= 1, dropout=0.0):
        """
        Args:
            time_emb_dim (int, optional): Dimension of time embedding dim for MLP.
                    If None, becomes identical to MONAI Class.
                    Defaults to None.
        """
        super().__init__()
        self.mlp = None
        self.conv1 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act=None,#"relu",
                norm=Norm.BATCH,
                dropout=dropout,
            )
        self.act = nn.ReLU()
        self.conv2 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            )
        self.time_emb_dim = time_emb_dim
        self.in_channels = in_channels
        if time_emb_dim is not None:
            self.mlp =nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_channels * 2))

    def forward(self, x, time_emb=None):
        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb)#.view(-1,self.time_emb_dim*self.in_channels))
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (scale + 1) + shift
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        return h
    

import numpy as np
import random


# set_seed(seed)
# c_og = ConvBlock(2,2,3)
# set_seed(seed)
# c = ConvBlock_with_time(2,2,3)
# set_seed(seed)
# c_time = ConvBlock_with_time(2,5,3,time_emb_dim=128)
# # set_seed(seed)
# in_x = torch.rand(10,5,8,8)
# time_x = torch.zeros(10,5,1)
# m = SinusoidalPositionEmbeddings(64)
# time_enc = m(time_x).squeeze()#(10,3)))
# lin = nn.Linear(64,128)
# time_enc = lin(time_enc)
# print("lol")
# print(time_enc.shape)
# c_time(in_x, time_emb=time_enc)

# out_og = c_og(in_x)
# out1 = c(in_x)
# out2 = c(in_x, time_emb=time_x)
# out3 = c_time(in_x, time_emb=time_x)
# out4 = c_time(in_x)

# print((out1 - out_og).mean())
# print((out1 - out2).mean())
# print((out1 - out3).mean())
# print((out1 - out4).mean())
# print((out3 - out4).mean())
# print(out1.shape)
    



# TODO: 
class UpConv_with_time(nn.Module):
    """
        Adapted from Monai code and https://huggingface.co/blog/annotated-diffusion.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3, strides=2, dropout=0.0,
                 time_emb_dim=None):
        """
        Args:
            time_emb_dim (int, optional): Dimension of time embedding dim for MLP.
                    If None, becomes identical to MONAI Class.
                    Defaults to None.
        """
        super().__init__()
        self.mlp = None
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act=None,#"relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )
        self.act = nn.ReLU()
        self.time_emb_dim = time_emb_dim
        self.in_channels = in_channels
        if time_emb_dim is not None:
            self.mlp =nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_channels * 2))

    def forward(self, x, time_emb=None):
        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb)#.view(-1,self.time_emb_dim*self.in_channels))
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (scale + 1) + shift
        x_u = self.up(x)
        x_u = self.act(x_u)
        return x_u



class AttentionLayer_with_time(nn.Module):
    """
        Adapted from Monai code and https://huggingface.co/blog/annotated-diffusion.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, 
                submodule,
                up_kernel_size=3,strides=2,dropout=0.0, time_emb_dim=None):
        """
        Args:
            time_emb_dim (int, optional): Dimension of time embedding dim for MLP.
                    If None, becomes identical to MONAI Class.
                    Defaults to None.
        """
        super().__init__()

        self.attention = AttentionBlock_with_time(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2,
            time_emb_dim=time_emb_dim
        )

        self.upconv = UpConv_with_time(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
            time_emb_dim=time_emb_dim,
        )

        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )

        self.submodule = submodule

    def forward(self, x, time_emb=None):
        inter = self.submodule(x,time_emb=time_emb)
        fromlower = self.upconv(inter, time_emb=time_emb)
        # fromlower = self.upconv(self.submodule(x,time_emb=time_emb), time_emb=time_emb)

        att = self.attention(g=fromlower, x=x, time_emb=time_emb)
        att_m = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m

# TODO: 
class AttentionBlock_with_time(nn.Module):
    """
        Adapted from Monai code and https://huggingface.co/blog/annotated-diffusion.
    """
    def __init__(self, spatial_dims, f_int, f_g, f_l, dropout=0.0, time_emb_dim=None):
        """
        Args:
            time_emb_dim (int, optional): Dimension of time embedding dim for MLP.
                    If None, becomes identical to MONAI Class.
                    Defaults to None.
        """
        super().__init__()
        self.mlp = None
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()
        self.time_emb_dim = time_emb_dim
        self.in_channels = f_l
        if time_emb_dim is not None:
            self.mlp =nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, f_l * 2))

    def forward(self, g, x, time_emb=None):
        if (self.mlp is not None) and (time_emb is not None):
            # time_emb = self.mlp(time_emb)
            time_emb = self.mlp(time_emb)#.view(-1,self.time_emb_dim*self.in_channels))
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (scale + 1) + shift

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        #TODO if problem.. might try : x_shifted = x * (scale + 1) + shift and keep x clean for the return...
        return x * psi




class Sequential_with_time(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input, time_emb):
        """
        Args:
            time_emb (torch.tensor): Allows to sequentially input both arguments to a module.
        """
        for module in self:
            input = module(input, time_emb)
        return input


class AttentionUnet_with_time(nn.Module):
    """
    Adapted from Monai code 
        and https://huggingface.co/blog/annotated-diffusion.
        and https://arxiv.org/abs/1804.03999
     Actual model proposed: attention UNet with additional time conditioning.
    """
    def __init__(self, spatial_dims, in_channels,out_channels,channels, strides,
                kernel_size= 3, up_kernel_size= 3, dropout= 0.0,
                time_encoding_dim=None):
        """
        Args:
            time_encoding_dim (int, optional): Dimension of time encoding for sinusoidal embedding.
                    Also specifies MLP size.
                    If None, becomes identical to MONAI Class.
                    Defaults to None.
        """
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.time_encoding_dim = time_encoding_dim
        self.time_emb_dim = time_encoding_dim*4 if time_encoding_dim is not None else None

        #TODO maybe keep head normal.....
        head = ConvBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=channels[0],
                                  dropout=dropout)
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels, strides) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer_with_time(spatial_dims=spatial_dims,in_channels=channels[0],out_channels=channels[1],
                                        submodule=Sequential_with_time(#nn.Sequential(
                                            ConvBlock_with_time(
                                                spatial_dims=spatial_dims,in_channels=channels[0],out_channels=channels[1],
                                                strides=strides[0],dropout=self.dropout,
                                                time_emb_dim=self.time_emb_dim
                                            ),
                                            subblock,),
                                        up_kernel_size=self.up_kernel_size,
                                        strides=strides[0],
                                        dropout=dropout,
                                        time_emb_dim=self.time_emb_dim
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.head = head
        self.encdec = encdec
        self.reduce_channels = reduce_channels

        # time embeddings
        # time_dim = dim * 4
        self.time_mlp = None
        self.channel_time_mlp = None
        if time_encoding_dim is not None:
            self.time_mlp = nn.Sequential(
                            SinusoidalPositionEmbeddings(self.time_encoding_dim),
                            nn.Linear(self.time_encoding_dim, self.time_emb_dim),
                            nn.GELU(),
                        )
            self.channel_time_mlp = nn.Linear(self.time_emb_dim*self.in_channels, self.time_emb_dim)
    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer_with_time(spatial_dims=self.dimensions,in_channels=in_channels,out_channels=out_channels,
                            submodule=ConvBlock_with_time(
                                spatial_dims=self.dimensions,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                dropout=self.dropout,
                                time_emb_dim=self.time_emb_dim
                            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
            time_emb_dim=self.time_emb_dim
        )

    def forward(self, x, time_x=None):
        time_emb = None
        if (self.time_mlp is not None) and (time_x is not None):
            time_emb = self.time_mlp(time_x).view(-1,self.time_emb_dim*self.in_channels)
            time_emb = self.channel_time_mlp(time_emb)
            # print(f"{time_emb.shape=}")
        x = self.head(x)
        x = self.encdec(x, time_emb)
        x = self.reduce_channels(x)
        return x
    

####### Compatibility tests:

# from monai.networks.nets import RegUNet, BasicUNet, UNETR, UNet, SegResNet, AttentionUnet

# seed =0 
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
# set_seed(seed)

# model = AttentionUnet(
#     spatial_dims=2, 
#     in_channels=4, 
#     out_channels=3,
#     channels=(16, 32, 64),
#     strides=(2, 2, 2, ),
#     kernel_size=3,
#     up_kernel_size=3, 
#     dropout=0.0)
# set_seed(seed)
# model_time_nothing = AttentionUnet_with_time(
#     spatial_dims=2, 
#     in_channels=4, 
#     out_channels=3,
#     channels=(16, 32, 64),
#     strides=(2, 2, 2, ),
#     kernel_size=3,
#     up_kernel_size=3, 
#     dropout=0.0,
#     time_encoding_dim=None)

# model_time = AttentionUnet_with_time(
#     spatial_dims=2, 
#     in_channels=4, 
#     out_channels=3,
#     channels=(16, 32, 64),
#     strides=(2, 2, 2, ),
#     kernel_size=3,
#     up_kernel_size=3, 
#     dropout=0.0,
#     time_encoding_dim=32)

# b = 7
# in_x = torch.rand(b,4,64,64)
# time_x = torch.zeros(b,4,1)

# in_x = torch.rand(10,5,8,8)
# time_x = torch.zeros(10,5,1)
# m = SinusoidalPositionEmbeddings(64)
# time_enc = m(time_x).squeeze()#(10,3)))
# lin = nn.Linear(64,128)

# out_time = model_time(in_x, time_x)
# # out2 = c(in_x, time_emb=time_x)
# # out3 = c_time(in_x, time_emb=time_x)
# # out4 = c_time(in_x)

# print((out1 - out_og).mean())
# print(out1.shape)
# print((out1 - out_time).mean())
# print(out1.shape)
# print(out_time.shape)
# # print((out1 - out2).mean())
# # print((out1 - out3).mean())

