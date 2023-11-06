from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from .nets import SwinTransformer

rearrange, _ = optional_import("einops", name="rearrange")
class Conv_decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        out_channels: int,
    ) -> None:
        super(Conv_decoder, self).__init__()
        if dim == 768:
            self.conv = nn.Sequential(
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 16, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    )
        elif dim == 384:
            self.conv = nn.Sequential(
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 8, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    )
        elif dim == 192:
            self.conv = nn.Sequential(
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 4, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    
                    )
        elif dim == 96:
            self.conv = nn.Sequential(
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        nn.ConvTranspose3d(dim // 2, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    )
        elif dim == 48:
            self.conv = nn.Sequential(
                        nn.ConvTranspose3d(dim, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    )
    def forward(self,x):
        return self.conv(x)

class SSL_multi_Head(nn.Module):
    # def __init__(self, args, upsample="vae", dim=768):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,dim=768,upsample="deconv"
    ) -> None:
        super(SSL_multi_Head, self).__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, out_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.decoder = nn.ModuleList(Conv_decoder(dim,out_channels=out_channels) for dim in [48,192,768])
            
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())
        
        x_rec0 = self.decoder[0](x_out[0])
        x_rec1 = self.decoder[1](x_out[2])
        x_rec2 = self.decoder[2](x_out[4])
        return x_rec0,x_rec1,x_rec2


class SSLHead(nn.Module):
    # def __init__(self, args, upsample="vae", dim=768):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,dim=768,upsample="deconv"
    ) -> None:
        super(SSLHead, self).__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, out_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]

        x_rec = self.conv(x_out)
        return x_rec
