# -*- coding: utf-8 -*-
import torch
from torch import nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import to_2tuple

try:
    from .pos_embed import get_2d_sincos_pos_embed
except ImportError:
    from pos_embed import get_2d_sincos_pos_embed


class ViT(nn.Module):
    """docstring for ViT"""

    def __init__(self, img_size: tuple or int = 224, patch_size: int = 16, in_chans: int = 3,
                 out_chans: int = 3, embed_dim: int = 1024, depth: int = 24, num_heads: int = 16,
                 mlp_ratio: float = 4., norm_layer: nn = nn.LayerNorm) -> None:
        super().__init__()
        self.in_chans, self.img_size, self.out_chans = in_chans, to_2tuple(img_size), out_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                  mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer,)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.encoder_pred = nn.Linear(embed_dim, patch_size**2 * out_chans, bias=True)

    def initialize_weights(self):
        # initialization initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: object) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h, w = self.img_size[0] // p, self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))

    def _add_cls_tokens(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

    def _add_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x + self.pos_embed[:, 1:, :]

    def forward(self, x):
        # add pos embed w/o cls token
        x = self._add_pos_embed(x)

        # append cls token
        x = self._add_cls_tokens(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.encoder_pred(self.norm(x))

        # remove cls token
        return self.unpatchify(x[:, 1:, :])


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return ViT(patch_size=16, embed_dim=768,
               depth=12, num_heads=12, mlp_ratio=4,
               norm_layer=partial(nn.LayerNorm, eps=1e-6),
               **kwargs)


def mae_vit_large_patch16_dec512d8b(**kwargs):
    return ViT(patch_size=16, embed_dim=1024,
               depth=24, num_heads=16, mlp_ratio=4,
               norm_layer=partial(nn.LayerNorm, eps=1e-6),
               **kwargs)


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    return ViT(patch_size=14, embed_dim=1280,
               depth=32, num_heads=16,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
               **kwargs)


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    model = mae_vit_base_patch16(img_size=(1344, 448), in_chans=192)
    res = []

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    model = model.cuda()
    left_img = torch.rand(1, 192, 1344, 448).cuda()
    for _ in range(100):
        res = model(left_img)
        print(res.shape)

    # image = model.unpatchify(res[1])
    # print(image.shape)
