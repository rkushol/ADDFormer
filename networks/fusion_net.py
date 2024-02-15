#Partial implementation taken from https://github.com/Beckschen/TransUNet
#Partial implementation taken from https://github.com/raoyongming/GFNet

import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from .vit_seg_modeling import VisionTransformer as ViT_seg, Block
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .gfnet import GFNet, GFNetPyramid
from functools import partial


class FusionNet(nn.Module):
    def __init__(self, args):
        super(FusionNet, self).__init__()

        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip

        self.num_classes = args.num_classes

        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

        self.vit = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        if args.gf_name == 'gfnet-xs':
            self.gf = GFNet(
                img_size=args.img_size,
                patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif args.gf_name == 'gfnet-ti':
            self.gf = GFNet(
                img_size=args.img_size,
                patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif args.gf_name == 'gfnet-s':
            self.gf = GFNet(
                img_size=args.img_size,
                patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif args.gf_name == 'gfnet-b':
            self.gf = GFNet(
                img_size=args.img_size,
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif args.gf_name == 'gfnet-h-ti':
            self.gf = GFNetPyramid(
                img_size=args.img_size,
                patch_size=4, embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
                mlp_ratio=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,
            )
        elif args.gf_name == 'gfnet-h-s':
            self.gf = GFNetPyramid(
                img_size=args.img_size,
                patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 10, 3],
                mlp_ratio=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.2, init_values=1e-5
            )
        elif args.gf_name == 'gfnet-h-b':
            self.gf = GFNetPyramid(
                img_size=args.img_size,
                patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 27, 3],
                mlp_ratio=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.4, init_values=1e-6
            )
        else:
            self.gf = None
            raise NotImplementedError

        if args.is_pretrain:

            self.vit.load_from(weights=np.load(config_vit.pretrained_path))

            checkpoint = torch.load(args.gf_path)

            checkpoint_model = checkpoint['model']
            state_dict = self.gf.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")

            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]

            if args.gf_name in ['gfnet-ti', 'gfnet-xs', 'gfnet-s', 'gfnet-b']:
                num_patches = (args.img_size // 16) ** 2
            elif args.gf_name in ['gfnet-h-ti', 'gfnet-h-s', 'gfnet-h-b']:
                num_patches = (args.img_size // 4) ** 2
            else:
                raise NotImplementedError

            num_extra_tokens = 0
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)

            scale_up_ratio = new_size / orig_size
            # class_token and dist_token are kept unchanged
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model['pos_embed'] = pos_tokens

            for name in checkpoint_model.keys():
                if 'complex_weight' in name:
                    h, w, num_heads = checkpoint_model[name].shape[0:3]  # h, w, c, 2
                    origin_weight = checkpoint_model[name]
                    upsample_h = h * new_size // orig_size
                    upsample_w = upsample_h // 2 + 1
                    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
                    new_weight = torch.nn.functional.interpolate(origin_weight, size=(upsample_h, upsample_w),
                                                                 mode='bicubic', align_corners=True).permute(0, 2, 3,
                                                                                                             1).reshape(
                        upsample_h, upsample_w, num_heads, 2)
                    checkpoint_model[name] = new_weight
            self.gf.load_state_dict(checkpoint_model, strict=True)

        self.vit_adapter = nn.Linear(768, 768)
        self.gf_adapter = nn.Linear(512, 768)

        self.position_embeddings = nn.Parameter(torch.zeros(1, 14 * 14 * 2, config_vit.hidden_size))

        self.fuser = Fuser(config_vit)

        self.head = Linear(config_vit.hidden_size, config_vit.n_classes)

    def forward(self, x, labels=None):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        _, vit_feature = self.vit(x)
        _, gf_feature = self.gf(x)

        vit_feature = self.vit_adapter(vit_feature)
        gf_feature = self.gf_adapter(gf_feature)

        fuse_feature = torch.cat([vit_feature, gf_feature], 1) + self.position_embeddings

        fuse_feature = self.fuser(fuse_feature)

        logits = self.head(fuse_feature.mean(1))

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits


class Fuser(nn.Module):
    def __init__(self, config, vis=False):
        super(Fuser, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_fusion_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded
