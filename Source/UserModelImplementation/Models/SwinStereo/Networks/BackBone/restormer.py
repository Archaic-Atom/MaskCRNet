# -*- coding: utf-8 -*-
# from pdb import set_trace as stx
import torch
from torch import nn
try:
    from .transformer_basic import OverlapPatchEmbed, TransformerBlock
    from .transformer_basic import Upsample, Downsample
    from .mask_reconstruction import reconstruct_img
except ImportError:
    from transformer_basic import OverlapPatchEmbed, TransformerBlock
    from transformer_basic import Upsample, Downsample
    from mask_reconstruction import reconstruct_img


class Restormer(nn.Module):
    def __init__(self, inp_channels: int = 3, out_channels: int = 3, dim: int = 48,
                 num_blocks: list = [4, 6, 6, 8], num_refinement_blocks: int = 4,
                 heads: list = [1, 2, 4, 8], ffn_expansion_factor: float = 2.66, bias: bool = False,
                 LayerNorm_type: str = 'WithBias',      # Other option 'BiasFree'
                 dual_pixel_task: bool = True,          # True for dual-pixel defocus deblurring only.
                 pre_train_opt: bool = False) -> None:  # Also set inp_channels=6
        super().__init__()
        self.dual_pixel_task, self.pre_train_opt = dual_pixel_task, pre_train_opt

        self.patch_embed, self.encoder_level1 = OverlapPatchEmbed(inp_channels, dim), \
            self._make_layer(
            dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[0])
        self.down1_2, self.encoder_level2 = Downsample(dim), self._make_layer(
            dim * 2**1, heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1])
        self.down2_3, self.encoder_level3 = Downsample(int(dim * 2**1)), self._make_layer(
            dim * 2**2, heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2])

        if self.pre_train_opt:
            self.down3_4, self.latent = Downsample(int(dim * 2**2)), self._make_layer(
                dim * 2**3, heads[3], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[3])
            self.up4_3, self.reduce_chan_level3, self.decoder_level3 = Upsample(int(dim * 2**3)), \
                nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias), self._make_layer(
                dim * 2**2, heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2])
            self.up3_2, self.reduce_chan_level2, self.decoder_level2 = Upsample(int(dim * 2**2)),\
                nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias), self._make_layer(
                dim * 2**1, heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1])
            self.up2_1, self.decoder_level1 = Upsample(int(dim * 2**1)), self._make_layer(
                dim * 2**1, heads[0], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[0])
            self.refinement = self._make_layer(
                dim * 2**1, heads[0], ffn_expansion_factor, bias, LayerNorm_type, num_refinement_blocks)

            if self.dual_pixel_task:
                self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
            self.output = nn.Conv2d(
                int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    @staticmethod
    def _make_layer(dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool,
                    LayerNorm_type: str, block_num: int) -> nn.Module:
        return nn.Sequential(*[TransformerBlock(
            dim=int(dim), num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(block_num)])

    @staticmethod
    def upsample(x: torch.Tensor, feat: torch.Tensor, upsample_func: object,
                 reduce_channle_func: object, decoder_func: object) -> torch.Tensor:
        x = upsample_func(x)
        x = torch.cat([x, feat], 1)
        x = reduce_channle_func(x) if reduce_channle_func is not None else x
        x = decoder_func(x)
        return x

    @staticmethod
    def bnchw2bchw(mask_img_patch: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = mask_img_patch.shape
        return mask_img_patch.reshape(b * n, c, h, w)

    @staticmethod
    def bchw2bnchw(mask_img_patch: torch.Tensor,
                   batch_size: int, patch_num: int) -> torch.Tensor:
        _, c, h, w = mask_img_patch.shape
        return mask_img_patch.reshape(batch_size, patch_num, c, h, w)

    @staticmethod
    def downsample(x: torch.Tensor, downsample_func: object, encoder_func: object) -> torch.Tensor:
        x_skip = downsample_func(x)
        x = encoder_func(x_skip)
        return x_skip, x

    def convert_dim(self, x: torch.Tensor, b: int, n: int,
                    h: int, w: int, random_sample_list: torch.Tensor) -> torch.Tensor:
        x = self.bchw2bnchw(x, b, n)
        _, _, c, _, _ = x.shape
        return reconstruct_img((b, c, h, w), x, random_sample_list)

    def forward(self, inp_img: torch.Tensor,
                source_img: torch.Tensor = None,
                random_sample_list: torch.Tensor = None) -> torch.Tensor:
        if self.pre_train_opt:
            b, _, h, w = source_img.shape
            _, n = random_sample_list.shape
            inp_img = self.bnchw2bchw(inp_img) if self.pre_train_opt else inp_img

        inp_enc_level1, out_enc_level1 = self.downsample(
            inp_img, self.patch_embed, self.encoder_level1)
        _, out_enc_level2 = self.downsample(out_enc_level1, self.down1_2, self.encoder_level2)
        _, out_enc_level3 = self.downsample(out_enc_level2, self.down2_3, self.encoder_level3)

        output, out_dec_level1, out_dec_level2 = None, inp_enc_level1, out_enc_level2

        if self.pre_train_opt:
            _, latent = self.downsample(out_enc_level3, self.down3_4, self.latent)

            inp_enc_level1 = self.convert_dim(
                inp_enc_level1, b, n, h, w, random_sample_list)
            out_enc_level1 = self.convert_dim(
                out_enc_level1, b, n, h, w, random_sample_list)
            out_enc_level2 = self.convert_dim(
                out_enc_level2, b, n, h // 2, w // 2, random_sample_list)
            out_enc_level3 = self.convert_dim(
                out_enc_level3, b, n, h // 4, w // 4, random_sample_list)
            latent = self.convert_dim(latent, b, n, h // 8, w // 8, random_sample_list)

            out_dec_level3 = self.upsample(
                latent, out_enc_level3, self.up4_3, self.reduce_chan_level3, self.decoder_level3)
            out_dec_level2 = self.upsample(
                out_dec_level3, out_enc_level2, self.up3_2, self.reduce_chan_level2,
                self.decoder_level2)
            out_dec_level1 = self.upsample(
                out_dec_level2, out_enc_level1, self.up2_1, None, self.decoder_level1)

            out_dec_level1 = self.refinement(out_dec_level1)

            if self.dual_pixel_task:
                output = self.output(out_dec_level1 + self.skip_conv(inp_enc_level1))
            else:
                output = self.output(output) + inp_img
        return output, out_dec_level1, out_dec_level2, out_enc_level3
