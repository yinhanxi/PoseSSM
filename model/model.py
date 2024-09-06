import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

from model.block.gcn_conv import GCN_module
from model.block.transformer import Transformer_module
from mamba_ssm import Mamba

    
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
    
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,    norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pose_gcn = GCN_module(dim,dim*2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.pose_transformer = Transformer_module(dim, num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1,attn_drop=attn_drop,drop_path=drop_path, norm_layer=nn.LayerNorm,length=length)
        self.pose_ssm = PoseSSM_module(dim, num_heads,  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

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