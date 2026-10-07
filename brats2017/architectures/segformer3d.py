import torch
import math
import copy
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List


def build_segformer3d_model(config=None):
    model = SegFormer3D(
        in_channels=config["model_parameters"]["in_channels"],
        sr_ratios=config["model_parameters"]["sr_ratios"],
        embed_dims=config["model_parameters"]["embed_dims"],
        patch_kernel_size=config["model_parameters"]["patch_kernel_size"],
        patch_stride=config["model_parameters"]["patch_stride"],
        patch_padding=config["model_parameters"]["patch_padding"],
        mlp_ratios=config["model_parameters"]["mlp_ratios"],
        num_heads=config["model_parameters"]["num_heads"],
        depths=config["model_parameters"]["depths"],
        agent_num_per_stage=config["model_parameters"].get(
            "agent_num_per_stage", [8, 8, 8, 8]
        ),
        agent_use_dwc=config["model_parameters"].get("agent_use_dwc", True),
        agent_register_num_per_stage=config["model_parameters"].get(
            "agent_register_num_per_stage", [0, 0, 4, 4]
        ),
        decoder_head_embedding_dim=config["model_parameters"][
            "decoder_head_embedding_dim"
        ],
        num_classes=config["model_parameters"]["num_classes"],
        decoder_dropout=config["model_parameters"]["decoder_dropout"],
    )
    return model


class SegFormer3D(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            sr_ratios: list = [4, 2, 1, 1],
            embed_dims: list = [32, 64, 160, 256],
            patch_kernel_size: list = [7, 3, 3, 3],
            patch_stride: list = [4, 2, 2, 2],
            patch_padding: list = [3, 1, 1, 1],
            mlp_ratios: list = [4, 4, 4, 4],
            num_heads: list = [1, 2, 5, 8],
            depths: list = [2, 2, 2, 2],
            agent_num_per_stage: list = [8, 8, 8, 8],
            agent_use_dwc: bool = True,
            agent_register_num_per_stage: list = [0, 0, 4, 4],
            decoder_head_embedding_dim: int = 128,
            num_classes: int = 3,
            decoder_dropout: float = 0.0,
    ):
        """
        in_channels: number of the input channels
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratios: at which rate increases the projection dim of the hidden_state in the mlp
        num_heads: number of attention heads
        depths: number of attention layers
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channel of the network
        decoder_dropout: dropout rate of the concatenated feature maps
        """
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
            agent_num_per_stage=agent_num_per_stage,
            agent_use_dwc=agent_use_dwc,
            agent_register_num_per_stage=agent_register_num_per_stage,
        )
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self.segformer_encoder(x)
        c1, c2, c3, c4 = x[0], x[1], x[2], x[3]
        seg_logits, aux_outputs = self.segformer_decoder(c1, c2, c3, c4)
        if return_aux:
            return seg_logits, aux_outputs
        return seg_logits


# ----------------------------------------------------- encoder -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channel: int = 4,
            embed_dim: int = 768,
            kernel_size: int = 7,
            stride: int = 4,
            padding: int = 3,
    ):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches



class EfficientSelfAttention3D(nn.Module):
    """
    Original Efficient Self-Attention used in SegFormer3D.

    In this version, Stage 1 and Stage 2 keep this original attention.
    It reduces the spatial length of K/V when sr_ratio > 1, while Q keeps
    the original token resolution.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 8,
            sr_ratio: int = 1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)

        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def _get_kv_input(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.sr_ratio > 1:
            n = cube_root(N)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n).contiguous()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.sr_norm(x_)
            return x_
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        kv_input = self._get_kv_input(x)
        kv = (
            self.key_value(kv_input)
            .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0].contiguous(), kv[1].contiguous()

        attn = self.softmax((q @ k.transpose(-2, -1)) * self.scale)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class AgentAttention3D(nn.Module):
    """
    3D Agent Attention for SegFormer3D.

    This module replaces the original SegFormer3D spatial-reduction self-attention.
    It follows the two-step Agent Attention idea from the second file:
    1) pooled agent tokens aggregate information from K/V tokens;
    2) original query tokens receive information from agent tokens.

    Notes for this merged version:
    - Stage 3 and Stage 4 use this module by default.
    - sr_ratio is preserved.
    - agent_num defaults to 8, i.e. 2 x 2 x 2 pooled agent tokens.
    - register tokens are optional latent slots. They never enter the spatial
      token sequence, so cube-root reshaping remains unchanged.
    - The DWC branch reshape is fixed to avoid head/channel order errors.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 8,
            sr_ratio: int = 1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
            agent_num: int = 8,
            use_dwc: bool = True,
            register_num: int = 0,
    ):
        super().__init__()
        assert (
                embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.agent_num = agent_num
        self.use_dwc = use_dwc
        self.register_num = register_num

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

        pool_size = cube_root(agent_num)
        assert (
                pool_size ** 3 == agent_num
        ), "agent_num must be a perfect cube for 3D pooling, e.g. 8 or 27."
        self.pool = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))

        if register_num > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, register_num, embed_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        if use_dwc:
            self.dwc = nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=embed_dim,
            )
        else:
            self.dwc = nn.Identity()

    def _get_kv_input(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.sr_ratio > 1:
            n = cube_root(N)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n).contiguous()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.sr_norm(x_)
            return x_
        return x

    def _tokens_to_3d(self, tokens: torch.Tensor, spatial_size: int) -> torch.Tensor:
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, spatial_size, spatial_size, spatial_size).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        n = cube_root(N)

        q_raw = self.query(x)
        kv_input = self._get_kv_input(x)
        kv = (
            self.key_value(kv_input)
            .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0].contiguous(), kv[1].contiguous()

        q_3d = q_raw.permute(0, 2, 1).reshape(B, C, n, n, n).contiguous()
        agent_tokens = self.pool(q_3d).flatten(2).transpose(1, 2).contiguous()

        q = (
            q_raw
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        agent_tokens = (
            agent_tokens
            .reshape(B, self.agent_num, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        context_tokens = agent_tokens
        if self.register_tokens is not None:
            register_tokens = (
                self.register_tokens
                .expand(B, -1, -1)
                .reshape(B, self.register_num, self.num_heads, self.attention_head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            context_tokens = torch.cat([agent_tokens, register_tokens], dim=2)

        # Step 1: agent/register tokens aggregate global K/V information.
        context_attn = self.softmax((context_tokens * self.scale) @ k.transpose(-2, -1))
        context_attn = self.attn_dropout(context_attn)
        context_v = context_attn @ v

        # Step 2: original query tokens receive information from agent/register tokens.
        q_attn = self.softmax((q * self.scale) @ context_tokens.transpose(-2, -1))
        q_attn = self.attn_dropout(q_attn)
        out = q_attn @ context_v
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()

        # Depth-wise 3D convolution branch restores local feature diversity.
        if self.use_dwc:
            if self.sr_ratio > 1:
                nk = v.shape[-2]
                nk_cube = cube_root(nk)
                v_tokens = v.transpose(1, 2).reshape(B, nk, C).contiguous()
                v_3d = self._tokens_to_3d(v_tokens, nk_cube)
                v_3d = F.interpolate(
                    v_3d, size=(n, n, n), mode="trilinear", align_corners=False
                )
            else:
                v_tokens = v.transpose(1, 2).reshape(B, N, C).contiguous()
                v_3d = self._tokens_to_3d(v_tokens, n)
            out = out + self.dwc(v_3d).flatten(2).transpose(1, 2).contiguous()

        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            mlp_ratio: int = 2,
            num_heads: int = 8,
            sr_ratio: int = 2,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
            attention_type: str = "efficient",
            agent_num: int = 8,
            agent_use_dwc: bool = True,
            agent_register_num: int = 0,
    ):
        """
        attention_type:
            - "efficient": original Efficient Self-Attention in SegFormer3D
            - "agent": modified 3D Agent Attention

        In this modified model:
            Stage 1-2 -> "efficient"
            Stage 3-4 -> "agent"
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        if attention_type == "efficient":
            self.attention = EfficientSelfAttention3D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qkv_bias=qkv_bias,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
        elif attention_type == "agent":
            self.attention = AgentAttention3D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qkv_bias=qkv_bias,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                agent_num=agent_num,
                use_dwc=agent_use_dwc,
                register_num=agent_register_num,
            )
        else:
            raise ValueError(
                f"Unsupported attention_type: {attention_type}. "
                "Choose from ['efficient', 'agent']."
            )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            sr_ratios: list = [8, 4, 2, 1],
            embed_dims: list = [64, 128, 320, 512],
            patch_kernel_size: list = [7, 3, 3, 3],
            patch_stride: list = [4, 2, 2, 2],
            patch_padding: list = [3, 1, 1, 1],
            mlp_ratios: list = [2, 2, 2, 2],
            num_heads: list = [1, 2, 5, 8],
            depths: list = [2, 2, 2, 2],
            agent_num_per_stage: list = [8, 8, 8, 8],
            agent_use_dwc: bool = True,
            agent_register_num_per_stage: list = [0, 0, 4, 4],
    ):
        super().__init__()

        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        self.tf_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    sr_ratio=sr_ratios[0],
                    qkv_bias=True,
                    attention_type="efficient",
                    agent_num=agent_num_per_stage[0],
                    agent_use_dwc=agent_use_dwc,
                    agent_register_num=agent_register_num_per_stage[0],
                )
                for _ in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.tf_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    sr_ratio=sr_ratios[1],
                    qkv_bias=True,
                    attention_type="efficient",
                    agent_num=agent_num_per_stage[1],
                    agent_use_dwc=agent_use_dwc,
                    agent_register_num=agent_register_num_per_stage[1],
                )
                for _ in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.tf_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    sr_ratio=sr_ratios[2],
                    qkv_bias=True,
                    attention_type="agent",
                    agent_num=agent_num_per_stage[2],
                    agent_use_dwc=agent_use_dwc,
                    agent_register_num=agent_register_num_per_stage[2],
                )
                for _ in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.tf_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    sr_ratio=sr_ratios[3],
                    qkv_bias=True,
                    attention_type="agent",
                    agent_num=agent_num_per_stage[3],
                    agent_use_dwc=agent_use_dwc,
                    agent_register_num=agent_register_num_per_stage[3],
                )
                for _ in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []

        x = self.embed_1(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block1):
            x = blk(x)
        x = self.norm1(x)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x = self.embed_2(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block2):
            x = blk(x)
        x = self.norm2(x)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x = self.embed_3(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block3):
            x = blk(x)
        x = self.norm3(x)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x = self.embed_4(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block4):
            x = blk(x)
        x = self.norm4(x)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        return out


class _MLP(nn.Module):
    def __init__(self, in_feature: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        n = cube_root(N)
        x = x.transpose(1, 2).reshape(B, C, n, n, n).contiguous()
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


###################################################################################
@torch.jit.script
def cube_root(n: int) -> int:
    x = n ** (1.0 / 3.0)
    return int(x + 0.5)


###################################################################################
class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BSRM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        mid_channels = max(in_channels // 4, 32)

        self.seg_branch = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.boundary_branch = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.attention_conv = nn.Conv3d(mid_channels, 1, kernel_size=1)

        self.refine_block = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv3d(mid_channels, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv3d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        seg_feat = self.seg_branch(x)
        bnd_feat = self.boundary_branch(x)

        attention_map = torch.sigmoid(self.attention_conv(bnd_feat))
        refined_feat = seg_feat + attention_map * self.refine_block(seg_feat)

        seg_logits = self.seg_head(refined_feat)
        boundary_logits = self.boundary_head(bnd_feat)

        aux_outputs = {
            "boundary_logits": boundary_logits,
        }
        return seg_logits, aux_outputs


###################################################################################
# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.bn(x)
        return x


###################################################################################
class SegFormerDecoderHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
            self,
            input_feature_dims: list = [512, 320, 128, 64],
            decoder_head_embedding_dim: int = 128,
            num_classes: int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.bsrm = BSRM(
            in_channels=decoder_head_embedding_dim,
            num_classes=num_classes,
        )

        self.upsample_volume = nn.Upsample(
            scale_factor=4.0, mode="trilinear", align_corners=False
        )

    def forward(self, c1, c2, c3, c4):
        n, _, _, _, _ = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
            .contiguous()
        )
        _c4 = torch.nn.functional.interpolate(
            _c4,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
            .contiguous()
        )
        _c3 = torch.nn.functional.interpolate(
            _c3,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
            .contiguous()
        )
        _c2 = torch.nn.functional.interpolate(
            _c2,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
            .contiguous()
        )

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        fused_feature = self.dropout(_c)
        seg_logits, aux_outputs = self.bsrm(fused_feature)
        seg_logits = self.upsample_volume(seg_logits)
        aux_outputs["boundary_logits"] = self.upsample_volume(aux_outputs["boundary_logits"])
        return seg_logits, aux_outputs


###################################################################################
if __name__ == "__main__":
    input = torch.randint(
        low=0,
        high=255,
        size=(1, 4, 128, 128, 128),
        dtype=torch.float,
    )
    input = input.to("cuda:0")
    segformer3D = SegFormer3D().to("cuda:0")

    output = segformer3D(input)
    print("seg only:", output.shape)

    output, aux = segformer3D(input, return_aux=True)
    print("seg with aux:", output.shape)
    print("boundary:", aux["boundary_logits"].shape)
