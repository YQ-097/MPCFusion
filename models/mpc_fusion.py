import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.PCAM import PCAMBlock


_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        #print(B , N , self.num_heads, self.qk_dim // self.num_heads ,self.qk_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)

            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, y, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)

            k = self.k(y_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(y_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(y).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class CDAM_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Cross_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, y, H, W, relative_pos):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class MPC(nn.Module):
    def __init__(self, img_size=128, in_chans=3, num_classes=1000, embed_dims=[46,92], stem_channel=16, fc_dim=1280,
                 num_heads=[1,2], mlp_ratios=[3.6,3.6], qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 depths=[1,1], qk_ratio=1, sr_ratios=[8,4], dp=0.1):
        super().__init__()
        self.out_dict = {}

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)



        self.f_conv_3_1 = nn.Conv2d(16*4, 16*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.f_conv_3_2 = nn.Conv2d(16*2, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_conv_2_1 = nn.Conv2d(32*4, 32*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.f_conv_2_2 = nn.Conv2d(32*2, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_conv_1_1 = nn.Conv2d(64*4, 64*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.f_conv_1_2 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=True)


        self.ca_3 = PCAMBlock(16 * 2, reduction=4, kernel_size=3)
        self.ca_2 = PCAMBlock(32 * 2, reduction=4, kernel_size=3)
        self.ca_1 = PCAMBlock(64 * 2, reduction=4, kernel_size=3)
        
        #################### ir cnn ####################
        self.ir_conv1_1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.ir_relu1_1 = nn.GELU()
        self.ir_conv1_2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.ir_relu1_2 = nn.GELU()

        
        self.ir_conv2_1  = nn.Conv2d(stem_channel*2, stem_channel*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.ir_relu2_1  = nn.GELU()
        self.ir_conv2_2 = nn.Conv2d(stem_channel*2, stem_channel*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.ir_relu2_2 = nn.GELU()

        
        self.ir_conv3_1  = nn.Conv2d(stem_channel*4, stem_channel*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.ir_relu3_1  = nn.GELU()
        self.ir_conv3_2  = nn.Conv2d(stem_channel*4, stem_channel*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.ir_relu3_2  = nn.GELU()


        #################### vis cnn ####################
        self.vis_conv1_1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.vis_relu1_1 = nn.GELU()
        self.vis_conv1_2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.vis_relu1_2 = nn.GELU()

        self.vis_conv2_1 = nn.Conv2d(stem_channel*2, stem_channel*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.vis_relu2_1 = nn.GELU()
        self.vis_conv2_2 = nn.Conv2d(stem_channel*2, stem_channel*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.vis_relu2_2 = nn.GELU()

        self.vis_conv3_1 = nn.Conv2d(stem_channel*4, stem_channel*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.vis_relu3_1 = nn.GELU()
        self.vis_conv3_2 = nn.Conv2d(stem_channel*4, stem_channel*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.vis_relu3_2 = nn.GELU()


        #################### ir transformer ####################
        self.ir_patch_embed_a = PatchEmbed(
            img_size=img_size // 4, patch_size = 1, in_chans=stem_channel*8, embed_dim=embed_dims[0])
        self.ir_patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size = 1, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.ir_relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.ir_patch_embed_a.num_patches, self.ir_patch_embed_a.num_patches//sr_ratios[0]//sr_ratios[0])) #self.ir_patch_embed_a.num_patches//sr_ratios[0]//sr_ratios[0])
        self.ir_relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.ir_patch_embed_b.num_patches, self.ir_patch_embed_b.num_patches//sr_ratios[1]//sr_ratios[1])) #self.ir_patch_embed_b.num_patches//sr_ratios[1]//sr_ratios[1])
        
        ir_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ir_cur = 0
        self.ir_blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=ir_dpr[ir_cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        ir_cur += depths[0]
        self.ir_blocks_b = nn.ModuleList([
            CDAM_Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=ir_dpr[ir_cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        #################### vis transformer ####################
        self.vis_patch_embed_a = PatchEmbed(
            img_size=img_size//4, patch_size=1, in_chans=stem_channel*8, embed_dim=embed_dims[0])
        self.vis_patch_embed_b = PatchEmbed(
            img_size=img_size//4, patch_size=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])


        self.vis_relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.vis_patch_embed_a.num_patches, self.vis_patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))#self.vis_patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]
        self.vis_relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.vis_patch_embed_b.num_patches, self.vis_patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))#self.vis_patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]

        vis_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        vis_cur = 0
        self.vis_blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=vis_dpr[vis_cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        vis_cur += depths[0]
        self.vis_blocks_b = nn.ModuleList([
            CDAM_Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=vis_dpr[vis_cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        #################### decode ####################
        # decode部分一个CMT模块，后面与encode部分的

        self.patch_embed_b = PatchEmbed(
            img_size=img_size//4, patch_size=1, in_chans=embed_dims[1] * 2, embed_dim=embed_dims[1])


        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches, self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))#self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        #消融
        # 第一层cnn
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#size=input_size[2:]

        self.conv1_1_1 = nn.Conv2d(92+64, 78, kernel_size=3, stride=1, padding=1, bias=True)  # conv1_1 conv2_1 从底下往上排序 #步长调试时候结合图像大小选择 #通道数也要改
        self.conv1_1_2 = nn.Conv2d(78, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1_1 = nn.GELU()


        # 第二层cnn
        self.conv2_1_1 = nn.Conv2d(64+32, 48, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv2_1_2 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.GELU()


        self.conv2_2_1 = nn.Conv2d(64+32+32, 64, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv2_2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_2 = nn.GELU()

        # 第三层cnn
        self.conv3_1_1 = nn.Conv2d(16+32, 24, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_1_2 = nn.Conv2d(24, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_1 = nn.GELU()

        self.conv3_2_1 = nn.Conv2d(16+32+16, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_2_2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_2 = nn.GELU()

        self.conv3_3_1 = nn.Conv2d(16+32+16+16, 40, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_3_2 = nn.Conv2d(40, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_3 = nn.GELU()

        self.conv_out = nn.Conv2d(16, in_chans, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.relu_out = nn.GELU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def convert_pos(self, x, model):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""

        B, C, H, W = x.shape
        num_patches = (H // 4) * (W // 4)#4
        relative_pos_a = torch.randn(1, num_patches, num_patches // 8 // 8).to('cpu')
        relative_pos_b = torch.randn(2, num_patches, num_patches // 4 // 4).to('cpu')
        #print(model.ir_relative_pos_a.shape)
        for k, v in model.state_dict().items():
            if 'ir_relative_pos_a' in k and relative_pos_a.shape != model.ir_relative_pos_a.shape:
                relative_pos_a = resize_pos_embed(model.ir_relative_pos_a, relative_pos_a)
                #self.ir_relative_pos_a = relative_pos_a
                self.out_dict[k] = relative_pos_a
            elif 'ir_relative_pos_b' in k and relative_pos_b.shape != model.ir_relative_pos_b.shape:
                self.out_dict[k] = resize_pos_embed(model.ir_relative_pos_b, relative_pos_b)
            elif 'vis_relative_pos_a' in k and relative_pos_a.shape != model.vis_relative_pos_a.shape:
                self.out_dict[k] = resize_pos_embed(model.vis_relative_pos_a, relative_pos_a)
            elif 'vis_relative_pos_b' in k and relative_pos_b.shape != model.vis_relative_pos_b.shape:
                self.out_dict[k] = resize_pos_embed(model.vis_relative_pos_b, relative_pos_b)
            elif 'relative_pos_b' in k and relative_pos_b.shape != model.relative_pos_b.shape:
                self.out_dict[k] = resize_pos_embed(model.relative_pos_b, relative_pos_b)
        #print(self.out_dict)
        # return out_dict

    def forward_features_ir(self, x):#之后把需要的特征return出去和vis特征cat
        B = x.shape[0]

        x = self.ir_conv1_1(x)
        x = self.ir_relu1_1(x)
        t = x
        x = self.ir_conv1_2(x)
        x = self.ir_relu1_2(x)

        #融合层
        ir_f1 = torch.cat([x, t], 1)
        
        x = self.ir_conv2_1(torch.cat([x,t], 1))
        x = self.ir_relu2_1(x)
        t = x
        x = self.ir_conv2_2(x)
        x = self.ir_relu2_2(x)
        ir_f2 = torch.cat([x, t], 1)
        
        x = self.ir_conv3_1(torch.cat([x,t], 1))
        x = self.ir_relu3_1(x)
        t = x
        x = self.ir_conv3_2(x)
        x = self.ir_relu3_2(x)
        ir_f3 = torch.cat([x, t], 1)

        x = torch.cat([x,t], 1)
        
        x, (H, W) = self.ir_patch_embed_a(x)

        if self.out_dict != {}:
            for i, blk in enumerate(self.ir_blocks_a):
                x = blk(x, H, W, self.out_dict["ir_relative_pos_a"])
        else:
            for i, blk in enumerate(self.ir_blocks_a):
                x = blk(x, H, W, self.ir_relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        return x,ir_f1,ir_f2,ir_f3

    def forward_features_vis(self, x):
        B = x.shape[0]
        x = self.vis_conv1_1(x)
        x = self.vis_relu1_1(x)
        t = x
        x = self.vis_conv1_2(x)
        x = self.vis_relu1_2(x)
        vis_f1 = torch.cat([x, t], 1)

        x = self.vis_conv2_1(torch.cat([x,t], 1))
        x = self.vis_relu2_1(x)
        t = x
        x = self.vis_conv2_2(x)
        x = self.vis_relu2_2(x)
        vis_f2 = torch.cat([x, t], 1)

        x = self.vis_conv3_1(torch.cat([x,t], 1))
        x = self.vis_relu3_1(x)
        t = x
        x = self.vis_conv3_2(x)
        x = self.vis_relu3_2(x)
        vis_f3 = torch.cat([x, t], 1)

        x = torch.cat([x, t], 1)

        x, (H, W) = self.vis_patch_embed_a(x)
        if self.out_dict != {}:
            for i, blk in enumerate(self.vis_blocks_a):
                x = blk(x, H, W, self.out_dict["vis_relative_pos_a"])
        else:
            for i, blk in enumerate(self.vis_blocks_a):
                x = blk(x, H, W, self.vis_relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x,vis_f1,vis_f2,vis_f3

    def intra_domain_fusion(self, x, y):
        x, (H, W) = self.ir_patch_embed_b(x)
        y, (H, W) = self.vis_patch_embed_b(y)
        A = x
        B = y
        if self.out_dict != {}:
            for i, blk in enumerate(self.ir_blocks_b):
                x = blk(x, B, H, W, self.out_dict["ir_relative_pos_b"])
        else:
            for i, blk in enumerate(self.ir_blocks_b):
                x = blk(x, B, H, W, self.ir_relative_pos_b)


        if self.out_dict != {}:
            for i, blk in enumerate(self.vis_blocks_b):
                y = blk(y, A, H, W, self.out_dict["vis_relative_pos_b"])
        else:
            for i, blk in enumerate(self.vis_blocks_b):
                y = blk(y, A, H, W, self.vis_relative_pos_b)

        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # 加入域间融合，先可以直接卷积过去
        y = y.permute(0, 2, 1).reshape(B, C, H, W)  # 加入域间融合，先可以直接卷积过去
        return x, y

    def decode(self, A, B, ir_f1, ir_f2, ir_f3, vis_f1, vis_f2, vis_f3): #x_ ir #y vis
        x = torch.cat([A, B], 1)
        x, (H, W) = self.patch_embed_b(x)

        if self.out_dict != {}:
            for i, blk in enumerate(self.blocks_b):
                x = blk(x, H, W, self.out_dict["relative_pos_b"])
        else:
            for i, blk in enumerate(self.blocks_b):
                x = blk(x, H, W, self.relative_pos_b)

        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        #第一层
        f1 = self.ca_1(vis_f3, ir_f3)
        f1 = self.f_conv_1_1(f1)
        f1 = self.f_conv_1_2(f1)

        x1_1 = torch.cat([f1, x], 1)
        x1_1 = self.conv1_1_1(x1_1)
        x1_1 = self.conv1_1_2(x1_1)
        x1_1 = self.relu1_1(x1_1)

        #第二层
        f1 = self.upsample(f1)
        x1_1 = self.upsample(x1_1)
        f2 = self.ca_2(vis_f2,ir_f2)
        f2 = self.f_conv_2_1(f2)
        f2 = self.f_conv_2_2(f2)

        x2_1 = torch.cat([f1, f2], 1)

        x2_1 = self.conv2_1_1(x2_1)
        x2_1 = self.conv2_1_2(x2_1)
        x2_1 = self.relu2_1(x2_1)

        x2_2 = torch.cat([f2, x1_1, x2_1], 1)
        x2_2 = self.conv2_2_1(x2_2)
        x2_2 = self.conv2_2_2(x2_2)
        x2_2 = self.relu2_2(x2_2)

        #第三层
        f2 = self.upsample(f2)
        x2_1 = self.upsample(x2_1)
        x2_2 = self.upsample(x2_2)
        f3 = self.ca_3(vis_f1,ir_f1)
        f3 = self.f_conv_3_1(f3)
        f3 = self.f_conv_3_2(f3)

        x3_1 = torch.cat([f3, f2], 1)
        x3_1 = self.conv3_1_1(x3_1)
        x3_1 = self.conv3_1_2(x3_1)
        x3_1 = self.relu3_1(x3_1)

        x3_2 = torch.cat([x3_1, x2_1, f3], 1)
        x3_2 = self.conv3_2_1(x3_2)
        x3_2 = self.conv3_2_2(x3_2)
        x3_2 = self.relu3_2(x3_2)

        x3_3 = torch.cat([x3_1, x3_2, x2_2, f3], 1)
        x3_3 = self.conv3_3_1(x3_3)
        x3_3 = self.conv3_3_2(x3_3)
        x3_3 = self.relu3_3(x3_3)

        x = self.conv_out(x3_3)
        x = self.relu_out(x)

        return x

    def forward(self, x, y):
        x, ir_f1, ir_f2, ir_f3= self.forward_features_ir(x)
        y, vis_f1, vis_f2, vis_f3 = self.forward_features_vis(y)
        x, y = self.intra_domain_fusion(x, y)
        output = self.decode(x, y, ir_f1, ir_f2, ir_f3, vis_f1, vis_f2, vis_f3)
        return output




def resize_pos_embed(posemb, posemb_new):
    input_tensor = torch.unsqueeze(posemb, 1)
    x, x1, x2 = posemb_new.shape
    output_tensor = F.interpolate(input_tensor, size=(x1, x2), mode='bilinear')
    posemb = output_tensor.reshape(x, x1, x2)

    return posemb


