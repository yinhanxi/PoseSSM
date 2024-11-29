import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

from model.block.gcn_conv import GCN_module
from model.block.transformer import Transformer_module
from model.block.pose_ssm import PoseSSM_module
from mamba_ssm import Mamba
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,    norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pose_gcn = GCN_module(dim,dim*2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.pose_transformer = Transformer_module(dim, num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1,attn_drop=attn_drop,drop_path=drop_path, norm_layer=nn.LayerNorm,length=length)
        self.pose_ssm = PoseSSM_module(dim,drop_path=drop_path, norm_layer=nn.LayerNorm)

    def forward(self, x): 
        
        res = x
        x = self.pose_gcn(x)    
        x = x + res
        
        res = x
        x = self.pose_ssm(x)
        x = x + res
        
        res = x
        x = self.pose_transformer(x)
        x = x + res
        return x


class PoseSSM(nn.Module):
    def __init__(self, args, depth=3, embed_dim=160, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()
        
        depth, embed_dim, mlp_hidden_dim, length  = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        
        self.patch_embed = nn.Linear(2, embed_dim)
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        #dpr = [x.item() for x in torch.linspace(0.1, 0.2, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)        
        self.fcn = nn.Linear(embed_dim, 3)


    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        x = self.fcn(x)

        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x
