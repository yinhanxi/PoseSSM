import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

class PoseSSM_module(nn.Module):
    def __init__(self, dim, num_heads,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_ssm = norm_layer(dim)
        
        self.ssm = Mamba(
                        d_model=dim, # Model dimension d_model
                        d_state=4,  # SSM state expansion factor
                        d_conv=1,    # Local convolution width
                        expand=2,    # Block expansion factor
                        )
        self.norm_out = norm_layer(dim)
        self.hop_wise = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 11, 14, 10, 12, 15, 13, 16]
        self.graph_wise = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 10, 13, 15, 11, 14, 16]
        self.joints_left = [2, 5, 8, 10, 13, 15]
        self.joints_right = [1, 4, 7, 11, 14, 16]
        self.bp_embed = nn.Parameter(torch.zeros(1, 5, dim))

    def forward(self, x):
        res = x
        x_bpe = torch.cat([self.bp_embed[:,:3,:],self.bp_embed[:,:3,:],self.bp_embed[:,:3,:],self.bp_embed[:,:1,:],self.bp_embed[:,3:,:],self.bp_embed[:,:1,:],self.bp_embed[:,3:,:],self.bp_embed[:,3:,:]],dim=1)
        x_ssm = x[:, self.hop_wise, :] + x_bpe
        x_ssm = self.norm_ssm(x_ssm)
        x_ssm = self.ssm(x_ssm)
        x_ssm = x_ssm[:, self.graph_wise, :]
        x = res + self.drop_path(x_ssm)
        return x
