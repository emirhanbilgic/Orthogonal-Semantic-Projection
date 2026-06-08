#!/usr/bin/env python3
"""
Compute IMAGE-LEVEL AUROC + mIoU + Accuracy + mAP for all methods on MS COCO 2017.

Supports four dictionary strategies:
  S1 = Classes  (other class embeddings as dictionary)
  S2 = WordNet  (hypernyms + hyponyms + siblings)
  S3 = GPT-OSS  (GPT-generated visual concept dictionary)
  S4 = Gemini   (LLM-generated visual concept dictionary)

Shows results BEFORE (baseline, no OMP) and AFTER (with OMP) for each strategy.

Usage:
    python scripts/compute_image_level_auroc_coco.py --limit 100
    python scripts/compute_image_level_auroc_coco.py --strategies S3 --limit 100
    python scripts/compute_image_level_auroc_coco.py --strategies S1 S4 --methods legrad_clip gradcam_siglip
    python scripts/compute_image_level_auroc_coco.py --methods daam_daam --strategies S3 --limit 50
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess
import open_clip

from benchmark_segmentation import (
    batch_intersection_union,
    get_ap_scores,
    batch_pix_accuracy,
)

from sparse_encoding import (
    omp_sparse_residual,
    compute_map_for_embedding,
    wordnet_neighbors_configured,
)

try:
    from daam_segmentation import DAAMSegmenter
except Exception:
    DAAMSegmenter = None

try:
    from diffusers.models.attention_processor import Attention
    from daam.hook import UNetCrossAttentionLocator
    from daam.heatmap import RawHeatMapCollection
    from daam.utils import auto_autocast
except Exception:
    Attention = None
    UNetCrossAttentionLocator = None
    RawHeatMapCollection = None
    auto_autocast = None


COCO_CLASSES = [
    "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear",
    "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus",
    "cake", "car", "carrot", "cat", "cell phone", "chair", "clock", "couch", "cow", "cup",
    "dining table", "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee", "giraffe",
    "hair drier", "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop",
    "microwave", "motorcycle", "mouse", "orange", "oven", "parking meter", "person", "pizza",
    "potted plant", "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink",
    "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
    "surfboard", "teddy bear", "tennis racket", "tie", "toilet", "toothbrush", "traffic light",
    "train", "truck", "tv", "umbrella", "vase", "wine glass", "zebra"
]


# ── Method + model definitions ─────────────────────────────────────────────

MODELS = {
    'clip':   {'model_name': 'ViT-B-16',       'pretrained': 'laion2b_s34b_b88k', 'model_type': 'CLIP'},
    'siglip': {'model_name': 'ViT-B-16-SigLIP', 'pretrained': 'webli',            'model_type': 'SigLIP'},
    'daam':   {'model_name': None,               'pretrained': None,                'model_type': 'DAAM',
               'daam_model_id': 'Manojb/stable-diffusion-2-base'},
}

# Hyperparameters per (method, model, strategy) for MS COCO 2017.
# COCO strategies: S1=Classes, S2=WordNet, S3=GPT-OSS, S4=Gemini
HYPERPARAMS = {
    ('legrad', 'clip'): {
        'S1': (0.4, 13, 0.8),    'S2': (0.425, 27, 0.75), 'S3': (0.4, 2, 0.55),    'S4': (0.425, 25, 0.75),
    },
    ('legrad', 'siglip'): {
        'S1': (0.275, 8, 0.9),   'S2': (0.425, 27, 0.75), 'S3': (0.425, 27, 0.85), 'S4': (0.275, 31, 0.8),
    },
    ('gradcam', 'clip'): {
        'S1': (0.125, 28, 0.75), 'S2': (0.15, 19, 0.65),  'S3': (0.15, 21, 0.75),  'S4': (0.15, 9, 0.75),
    },
    ('gradcam', 'siglip'): {
        'S1': (0.25, 18, 0.8),   'S2': (0.15, 15, 0.65),  'S3': (0.225, 24, 0.8),  'S4': (0.15, 19, 0.6),
    },
    ('chefercam', 'clip'): {
        'S1': (0.1, 14, 0.8),    'S2': (0.125, 11, 0.65), 'S3': (0.125, 17, 0.75), 'S4': (0.1, 18, 0.8),
    },
    ('chefercam', 'siglip'): {
        'S1': (0.1, 8, 0.85),    'S2': (0.1, 14, 0.85),   'S3': (0.1, 2, 0.85),    'S4': (0.1, 16, 0.9),
    },
    ('attentioncam', 'clip'): {
        'S1': (0.4, 20, 0.7),    'S2': (0.425, 14, 0.6),  'S3': (0.4, 12, 0.75),   'S4': (0.55, 19, 0.75),
    },
    ('attentioncam', 'siglip'): {
        'S1': (0.25, 12, 0.8),   'S2': (0.275, 22, 0.5),  'S3': (0.35, 18, 0.8),   'S4': (0.3, 6, 0.85),
    },
    ('daam', 'daam'): {
        'S1': (0.25, 2, 0.7),    'S2': (0.3, 27, 0.85),   'S3': (0.125, 7, 0.8),   'S4': (0.425, 2, 1.0),
    },
}

DAAM_OMP_BETA = {'S1': 0.5, 'S2': 0.1, 'S3': 0.1, 'S4': 0.1}

STRATEGY_NAMES = {
    'S1': 'Classes', 'S2': 'WordNet', 'S3': 'GPT-OSS', 'S4': 'Gemini',
}

DICT_STRATEGIES = ['S1', 'S2', 'S3', 'S4']

METHOD_MODEL_PAIRS = [
    ('legrad', 'clip'), ('legrad', 'siglip'),
    ('chefercam', 'clip'), ('chefercam', 'siglip'),
    ('attentioncam', 'clip'), ('attentioncam', 'siglip'),
    ('gradcam', 'clip'), ('gradcam', 'siglip'),
    ('daam', 'daam'),
]


def build_config(method, model_key, strategy):
    tau_act, atoms, tau_cos = HYPERPARAMS[(method, model_key)][strategy]
    cfg = {
        **MODELS[model_key],
        'method': method,
        'dict_strategy': strategy,
        'sparse_threshold': tau_act,
        'atoms': atoms,
        'max_dict_cos_sim': tau_cos,
    }
    if method == 'daam':
        cfg['omp_beta'] = DAAM_OMP_BETA[strategy]
    return cfg


def build_all_configs(strategies=None, methods=None):
    if strategies is None:
        strategies = DICT_STRATEGIES
    configs = {}
    for method, model_key in METHOD_MODEL_PAIRS:
        if methods and f"{method}_{model_key}" not in methods:
            continue
        for strat in strategies:
            name = f"{method}_{model_key}_{strat}"
            configs[name] = build_config(method, model_key, strat)
    return configs


# ── Heatmap computation functions ──────────────────────────────────────────

def compute_gradcam_heatmap(model, image, text_emb_1x, layer_index=8):
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    return heatmap[0, 0].clamp(0, 1).detach().cpu()


def compute_lrp_heatmap(model, image, text_emb_1x):
    """AttentionCAM heatmap. Verbatim from optimize_anti_hallucination.py."""
    import math
    from open_clip.timm_model import TimmModel

    H_img, W_img = image.shape[-2:]

    try:
        is_siglip = isinstance(model.visual, TimmModel)

        with torch.enable_grad():
            if is_siglip:
                pooler = model.visual.trunk.attn_pool
                blocks = list(model.visual.trunk.blocks)

                x = model.visual.trunk.patch_embed(image)
                if x.dim() == 4:
                    B, H, W, C = x.shape
                    x = x.reshape(B, H * W, C)
                else:
                    B, _, C = x.shape

                if model.visual.trunk.pos_embed is not None:
                    x = x + model.visual.trunk.pos_embed

                for block in blocks:
                    x = block(x)

                B, N, C = x.shape

                if pooler.pos_embed is not None:
                    x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)

                q_latent = pooler.latent.expand(B, -1, -1)
                q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
                kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                q, k = pooler.q_norm(q), pooler.k_norm(k)

                attn_probs = (q * pooler.scale) @ k.transpose(-2, -1)
                attn_probs = attn_probs.softmax(dim=-1)
                attn_probs.requires_grad_(True)

                x_pool = (attn_probs @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
                x_pool = pooler.proj(x_pool)
                x_pool = pooler.proj_drop(x_pool)
                x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))

                if pooler.pool == 'token':
                    pooled_feat = x_pool[:, 0]
                elif pooler.pool == 'avg':
                    pooled_feat = x_pool.mean(1)
                else:
                    pooled_feat = x_pool[:, 0]

                image_features = F.normalize(pooled_feat, dim=-1)

                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()

                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity, inputs=[attn_probs],
                    retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                cam = attn_probs * grad
                cam = cam.mean(dim=1)[:, 0]
                cam = cam.clamp(min=0)

                num_patches = cam.shape[-1]
                grid_size = int(math.sqrt(num_patches))

                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                heatmap = cam[0].reshape(grid_size, grid_size)

            else:
                image_features = model.encode_image(image, normalize=True)

                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()

                if hasattr(model.visual, 'transformer'):
                    blocks_list = list(model.visual.transformer.resblocks)
                else:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                last_block = blocks_list[-1]

                if not hasattr(last_block, 'attn') or not hasattr(last_block.attn, 'attention_maps'):
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                attn_map = last_block.attn.attention_maps

                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity, inputs=[attn_map],
                    retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                grad_weights = grad.mean(dim=[1, 2], keepdim=True)
                cam = attn_map * grad_weights
                cls_attn = cam[:, 0, 1:]
                cls_attn = cls_attn.mean(dim=0).clamp(min=0)

                num_patches = cls_attn.shape[0]
                grid_size = int(math.sqrt(num_patches))

                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                heatmap = cls_attn.reshape(grid_size, grid_size)

        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(H_img, W_img), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap.detach().cpu()

    except Exception:
        return torch.ones(H_img, W_img, device='cpu') * 0.5


def compute_chefercam(model, image, text_emb_1x):
    """GradCAM on last attention layer (attn_gradcam baseline). Verbatim from optimize_anti_hallucination.py."""
    import math
    from open_clip.timm_model import TimmModel

    model.zero_grad()
    num_prompts = text_emb_1x.shape[0]

    if isinstance(model.visual, TimmModel):
        blocks = list(model.visual.trunk.blocks)
        is_timm = True
    else:
        blocks = list(model.visual.transformer.resblocks)
        is_timm = False

    with torch.enable_grad():
        if is_timm:
            pooler = model.visual.trunk.attn_pool
            blocks = list(model.visual.trunk.blocks)

            x = model.visual.trunk.patch_embed(image)

            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H * W, C)
            else:
                B, _, C = x.shape

            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed

            for block in blocks:
                x = block(x)

            B, N, C = x.shape

            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)

            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)

            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            q, k = pooler.q_norm(q), pooler.k_norm(k)

            attn_weights = (q * pooler.scale) @ k.transpose(-2, -1)
            attn_weights = attn_weights.softmax(dim=-1)
            attn_weights.requires_grad_(True)

            x_pool = (attn_weights @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))

            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0]

            image_features = F.normalize(pooled_feat, dim=-1)
            bsz = B

        else:
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            batch_size = x.shape[0]
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)

            num_patches = x.shape[1] - 1
            if hasattr(model.visual, 'original_pos_embed'):
                pos_embed = model.visual.original_pos_embed
            else:
                pos_embed = model.visual.positional_embedding

            if pos_embed.shape[0] != x.shape[1]:
                cls_pos = pos_embed[:1]
                patch_pos = pos_embed[1:]
                orig_size = int(math.sqrt(patch_pos.shape[0]))
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                new_size = int(math.sqrt(num_patches))
                patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)

            x = x + pos_embed.unsqueeze(0).to(x.dtype)

            if hasattr(model.visual, 'ln_pre'):
                x = model.visual.ln_pre(x)

            x = x.permute(1, 0, 2)

            for i in range(len(blocks) - 1):
                x = blocks[i](x)

            last_block = blocks[-1]
            last_attn = last_block.attn
            x_normed = last_block.ln_1(x)

            qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            seq_len, bsz, embed_dim = q.shape
            num_heads = last_attn.num_heads
            head_dim = embed_dim // num_heads

            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)

            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights.requires_grad_(True)

            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = last_attn.out_proj(attn_output)

            x = x + attn_output
            x = x + last_block.mlp(last_block.ln_2(x))

            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)

        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)

        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False, allow_unused=True)[0]

        if grad is None:
            grad = torch.zeros_like(attn_weights)

        if is_timm:
            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)[:, 0]
            num_patches = cam.shape[-1]
        else:
            seq_len = attn_weights.shape[1]
            num_heads = blocks[-1].attn.num_heads
            grad = grad.view(bsz, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)

            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)
            cam = cam[:, 0, 1:]
            num_patches = cam.shape[-1]

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size

        heatmap = cam.reshape(bsz, 1, h, w)

        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap[0, 0].detach().cpu()


def compute_transformer_attribution(model, image, text_emb_1x, start_layer=1):
    """Transformer Attribution (full LRP). Verbatim from optimize_anti_hallucination.py."""
    import math
    from open_clip.timm_model import TimmModel

    model.zero_grad()

    if isinstance(model.visual, TimmModel):
        blocks = list(model.visual.trunk.blocks)
        is_timm = True
    else:
        blocks = list(model.visual.transformer.resblocks)
        is_timm = False

    num_layers = len(blocks)

    if start_layer < 0:
        start_layer = num_layers + start_layer
    start_layer = max(0, min(start_layer, num_layers - 1))

    num_prompts = text_emb_1x.shape[0]

    with torch.enable_grad():
        all_attn_weights = []

        if is_timm:
            x = model.visual.trunk.patch_embed(image)

            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H * W, C)

            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed

            pooler = model.visual.trunk.attn_pool

            B, N, C = x.shape

            for i, block in enumerate(blocks):
                if i < start_layer:
                    x = block(x)
                else:
                    x_normed = block.norm1(x)
                    attn = block.attn

                    qkv = attn.qkv(x_normed).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = attn.q_norm(q), attn.k_norm(k)

                    attn_w = (q @ k.transpose(-2, -1)) * attn.scale
                    attn_w = attn_w.softmax(dim=-1)
                    attn_w.requires_grad_(True)
                    all_attn_weights.append(attn_w)

                    attn_out = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
                    attn_out = attn.proj(attn_out)
                    attn_out = attn.proj_drop(attn_out)

                    if hasattr(block, 'ls1'):
                        attn_out = block.ls1(attn_out)

                    x = x + attn_out

                    x_mlp = block.mlp(block.norm2(x))
                    if hasattr(block, 'ls2'):
                        x_mlp = block.ls2(x_mlp)
                    x = x + x_mlp

            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)

            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = pooler.q_norm(q), pooler.k_norm(k)

            pool_attn = (q * pooler.scale) @ k.transpose(-2, -1)
            pool_attn = pool_attn.softmax(dim=-1)

            x_pool = (pool_attn @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))

            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0]

            image_features = F.normalize(pooled_feat, dim=-1)
            bsz = B

        else:
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            batch_size = x.shape[0]
            bsz = batch_size
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)

            num_patches = x.shape[1] - 1
            if hasattr(model.visual, 'original_pos_embed'):
                pos_embed = model.visual.original_pos_embed
            else:
                pos_embed = model.visual.positional_embedding

            if pos_embed.shape[0] != x.shape[1]:
                cls_pos = pos_embed[:1]
                patch_pos = pos_embed[1:]
                orig_size = int(math.sqrt(patch_pos.shape[0]))
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                new_size = int(math.sqrt(num_patches))
                patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)

            x = x + pos_embed.unsqueeze(0).to(x.dtype)

            if hasattr(model.visual, 'ln_pre'):
                x = model.visual.ln_pre(x)

            x = x.permute(1, 0, 2)

            for i, block in enumerate(blocks):
                if i < start_layer:
                    x = block(x)
                else:
                    attn_module = block.attn
                    x_normed = block.ln_1(x)

                    qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)

                    seq_len, bsz_tmp, embed_dim = q.shape
                    num_heads = attn_module.num_heads
                    head_dim = embed_dim // num_heads

                    q = q.contiguous().view(seq_len, bsz_tmp * num_heads, head_dim).transpose(0, 1)
                    k = k.contiguous().view(seq_len, bsz_tmp * num_heads, head_dim).transpose(0, 1)
                    v = v.contiguous().view(seq_len, bsz_tmp * num_heads, head_dim).transpose(0, 1)

                    scale = float(head_dim) ** -0.5
                    attn_w = torch.bmm(q * scale, k.transpose(-2, -1))
                    attn_w = F.softmax(attn_w, dim=-1)
                    attn_w.requires_grad_(True)
                    all_attn_weights.append(attn_w)

                    attn_output = torch.bmm(attn_w, v)
                    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz_tmp, embed_dim)
                    attn_output = attn_module.out_proj(attn_output)

                    x = x + attn_output
                    x = x + block.mlp(block.ln_2(x))

            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)

        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)

        grads = torch.autograd.grad(s, all_attn_weights, retain_graph=False, create_graph=False, allow_unused=True)

        layer_contributions = []
        for i, (grad, attn_w) in enumerate(zip(grads, all_attn_weights)):
            if grad is None:
                grad = torch.zeros_like(attn_w)

            num_heads = blocks[start_layer + i].attn.num_heads

            if grad.dim() == 3:
                grad = grad.view(bsz, num_heads, grad.shape[1], grad.shape[2])
                attn_w = attn_w.view(bsz, num_heads, attn_w.shape[1], attn_w.shape[2])

            grad = torch.clamp(grad, min=0)
            weighted_attn = grad * attn_w
            weighted_attn = weighted_attn.mean(dim=1)

            if is_timm:
                cls_to_patches = weighted_attn.max(dim=1).values
            else:
                cls_to_patches = weighted_attn[:, 0, 1:]

            layer_contributions.append(cls_to_patches)

        aggregated = sum(layer_contributions)

        num_patches = aggregated.shape[-1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size

        heatmap = aggregated.reshape(bsz, 1, h, w)

        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap[0, 0].detach().cpu()


# ── Key-Space OMP for DAAM ─────────────────────────────────────────────────

def get_token_indices(tokenizer, prompt: str, concept: str) -> List[int]:
    tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)

    indices = []
    concept_len = len(concept_tokens)
    for i in range(len(tokens) - concept_len + 1):
        if tokens[i:i + concept_len] == concept_tokens:
            indices.extend(range(i + 1, i + 1 + concept_len))

    if not indices:
        for i, tok in enumerate(tokens):
            if concept.lower() in tok.lower().replace('</w>', ''):
                indices.append(i + 1)

    return indices


class KeySpaceOMPProcessor:
    """Orthogonalizes target token's key vector against distractor keys in UNet cross-attention."""

    def __init__(self, target_token_indices, distractor_token_indices, beta=1.0,
                 heat_maps=None, layer_idx=0, latent_hw=4096, context_size=77,
                 parent_trace=None):
        self.target_token_indices = target_token_indices
        self.distractor_token_indices = distractor_token_indices
        self.beta = beta
        self.heat_maps = heat_maps
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw
        self.context_size = context_size
        self.parent_trace = parent_trace

    def _orthogonalize_keys(self, key, n_heads):
        key = key.clone()
        for target_idx in self.target_token_indices:
            target_key = key[:, target_idx, :]
            for dist_indices in self.distractor_token_indices:
                for dist_idx in dist_indices:
                    dist_key = key[:, dist_idx, :]
                    dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                    projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                    target_key = target_key - self.beta * projection
            key[:, target_idx, :] = target_key
        return key

    def _unravel_attn(self, x):
        import math
        factor = int(math.sqrt(self.latent_hw // x.shape[1]))
        if factor == 0:
            factor = 1
        hw = int(math.sqrt(x.shape[1]))
        maps = x.reshape(x.shape[0], hw, hw, x.shape[2])
        maps = maps.permute(0, 3, 1, 2)
        return maps

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        import math
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        is_cross_attention = (key.shape[1] == self.context_size)
        if is_cross_attention and self.beta > 0:
            key = self._orthogonalize_keys(key, attn.heads)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        factor = (int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
                  if attention_probs.shape[1] > 0 else 8)
        if self.parent_trace is not None:
            self.parent_trace._gen_idx += 1

        if (self.heat_maps is not None
                and attention_probs.shape[-1] == self.context_size
                and factor != 8):
            maps = self._unravel_attn(attention_probs)
            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def run_daam_with_key_space_omp(segmenter, image_pil, target_concept, competing_concepts,
                                beta=1.0, size=512):
    tokenizer = segmenter.tokenizer
    text_encoder = segmenter.text_encoder
    vae = segmenter.vae
    unet = segmenter.unet
    scheduler = segmenter.scheduler
    device = segmenter.device

    w, h = image_pil.size
    dtype = next(vae.parameters()).dtype

    img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = img_arr * 2.0 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * 0.18215

    all_concepts = [target_concept] + competing_concepts
    combined_prompt = f"a photo of a {', a '.join(all_concepts)}."

    target_indices = get_token_indices(tokenizer, combined_prompt, target_concept)
    distractor_indices = [
        get_token_indices(tokenizer, combined_prompt, comp)
        for comp in competing_concepts
    ]

    context_size = tokenizer.model_max_length
    target_indices = [i for i in target_indices if i < context_size]
    distractor_indices = [
        [i for i in group if i < context_size]
        for group in distractor_indices
    ]
    distractor_indices = [g for g in distractor_indices if g]

    if not target_indices:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)

    text_input = tokenizer(
        combined_prompt, padding="max_length",
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    scheduler.set_timesteps(50, device=device)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([21], device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep)

    heat_maps = RawHeatMapCollection()
    locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
    cross_attn_modules = locator.locate(unet)

    latent_hw = 4096

    class GenIdxTracker:
        def __init__(self):
            self._gen_idx = 0
    tracker = GenIdxTracker()

    original_processors = {}
    for idx, module in enumerate(cross_attn_modules):
        original_processors[idx] = module.processor
        module.set_processor(KeySpaceOMPProcessor(
            target_token_indices=target_indices,
            distractor_token_indices=distractor_indices,
            beta=beta,
            heat_maps=heat_maps,
            layer_idx=idx,
            latent_hw=latent_hw,
            context_size=77,
            parent_trace=tracker,
        ))

    try:
        with torch.no_grad():
            _ = unet(
                noisy_latents, timestep,
                encoder_hidden_states=text_embeddings
            ).sample
    finally:
        for idx, module in enumerate(cross_attn_modules):
            module.set_processor(original_processors[idx])

    x = int(np.sqrt(latent_hw))
    factors = {0, 1, 2, 4, 8, 16, 32, 64}

    all_merges = []
    with auto_autocast(dtype=torch.float32):
        for (factor, layer, head), heat_map in heat_maps:
            if factor in factors and factor != 8:
                heat_map = heat_map.unsqueeze(1)
                all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

    if not all_merges:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)

    maps = torch.stack(all_merges, dim=0)
    maps = maps.mean(0)[:, 0]

    target_maps = []
    for tidx in target_indices:
        if tidx < maps.shape[0]:
            target_maps.append(maps[tidx])

    if not target_maps:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)

    heatmap = torch.stack(target_maps).mean(0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.cpu()


# ── Helper functions ───────────────────────────────────────────────────────

def load_json_dictionary(path):
    with open(path, 'r') as f:
        return json.load(f)


def _concepts_from_json_dict(dictionary, class_name):
    entry = dictionary.get(class_name)
    if entry is None:
        return []
    return (
        entry.get('visual_confusers', []) +
        entry.get('co_occurring_context', []) +
        entry.get('semantic_hierarchy', [])
    )


def load_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    arr = np.array(mask)
    return (arr > 128).astype(np.uint8)


def build_sparse_embedding(text_emb_1x, target_class_name, config, ctx):
    """Build OMP sparse residual embedding for any dictionary strategy.

    ctx keys: tokenizer, model, device, all_text_embs, target_idx,
              unique_objects, gpt_dict, gemini_dict
    """
    strategy = config['dict_strategy']
    max_dict_cos_sim = config['max_dict_cos_sim']
    atoms = config['atoms']
    tokenizer = ctx['tokenizer']
    model = ctx['model']
    device = ctx['device']

    parts = []

    if strategy == 'S1':
        all_embs = ctx['all_text_embs']
        target_idx = ctx['target_idx']
        if target_idx > 0:
            parts.append(all_embs[:target_idx])
        if target_idx + 1 < all_embs.shape[0]:
            parts.append(all_embs[target_idx + 1:])

    elif strategy == 'S2':
        raw_neighbors = wordnet_neighbors_configured(
            target_class_name,
            use_synonyms=False,
            use_hypernyms=True,
            use_hyponyms=True,
            use_siblings=True,
            limit_per_relation=8,
        )
        if raw_neighbors:
            neighbor_prompts = [f"a photo of a {w}." for w in raw_neighbors]
            n_tok = tokenizer(neighbor_prompts).to(device)
            with torch.no_grad():
                n_emb = model.encode_text(n_tok)
                n_emb = F.normalize(n_emb, dim=-1)
            parts.append(n_emb)

    elif strategy == 'S3':
        concepts = _concepts_from_json_dict(ctx['gpt_dict'], target_class_name)
        if concepts:
            prompts = [f"a photo of a {c}." for c in concepts]
            c_tok = tokenizer(prompts).to(device)
            with torch.no_grad():
                c_emb = model.encode_text(c_tok)
                c_emb = F.normalize(c_emb, dim=-1)
            parts.append(c_emb)

    elif strategy == 'S4':
        concepts = _concepts_from_json_dict(ctx['gemini_dict'], target_class_name)
        if concepts:
            prompts = [f"a photo of a {c}." for c in concepts]
            c_tok = tokenizer(prompts).to(device)
            with torch.no_grad():
                c_emb = model.encode_text(c_tok)
                c_emb = F.normalize(c_emb, dim=-1)
            parts.append(c_emb)

    if parts:
        D = torch.cat(parts, dim=0)
        D = F.normalize(D, dim=-1)
        if 0.0 < max_dict_cos_sim < 1.0:
            sim = (D @ text_emb_1x.t()).squeeze(-1).abs()
            D = D[sim < max_dict_cos_sim]
    else:
        D = text_emb_1x.new_zeros((0, text_emb_1x.shape[-1]))

    return omp_sparse_residual(text_emb_1x, D, max_atoms=atoms)


def build_daam_competing_concepts(target_class_name, config, ctx):
    strategy = config['dict_strategy']
    atoms = config['atoms']

    all_competing = []

    if strategy == 'S1':
        for obj in ctx['unique_objects']:
            if obj.lower() != target_class_name.lower():
                all_competing.append(obj)

    elif strategy == 'S2':
        raw_neighbors = wordnet_neighbors_configured(
            target_class_name,
            use_synonyms=False, use_hypernyms=True,
            use_hyponyms=True, use_siblings=True,
            limit_per_relation=8,
        )
        all_competing.extend([n for n in raw_neighbors if n.lower() != target_class_name.lower()])

    elif strategy == 'S3':
        all_competing.extend(_concepts_from_json_dict(ctx['gpt_dict'], target_class_name))

    elif strategy == 'S4':
        all_competing.extend(_concepts_from_json_dict(ctx['gemini_dict'], target_class_name))

    seen = set()
    unique = []
    for c in all_competing:
        cl = c.lower()
        if cl not in seen and cl != target_class_name.lower():
            seen.add(cl)
            unique.append(c)

    if atoms > 0 and len(unique) > atoms:
        unique = unique[:atoms]
    return unique


def get_raw_heatmap(method, model, img_t, text_emb_1x,
                    daam_segmenter=None, base_img=None, target_class_name=None,
                    competing_concepts=None, omp_beta=1.0):
    if method == 'legrad':
        return compute_map_for_embedding(model, img_t, text_emb_1x)
    elif method == 'gradcam':
        return compute_gradcam_heatmap(model, img_t, text_emb_1x, layer_index=8)
    elif method == 'chefercam':
        return compute_transformer_attribution(model, img_t, text_emb_1x, start_layer=1)
    elif method == 'attentioncam':
        return compute_lrp_heatmap(model, img_t, text_emb_1x)
    elif method == 'daam':
        if competing_concepts:
            return run_daam_with_key_space_omp(
                daam_segmenter, base_img, target_concept=target_class_name,
                competing_concepts=competing_concepts, beta=omp_beta, size=512,
            )
        else:
            prompt_text = f"a photo of a {target_class_name}."
            return daam_segmenter.predict(base_img, prompt_text, size=512)
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_heatmap(heatmap, method, H_gt, W_gt):
    H_hm, W_hm = heatmap.shape[-2], heatmap.shape[-1]
    heatmap_resized = F.interpolate(
        heatmap.view(1, 1, H_hm, W_hm), size=(H_gt, W_gt),
        mode='bilinear', align_corners=False
    ).squeeze()
    if method == 'legrad':
        return heatmap_resized
    return (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)


def compute_seg_metrics(heatmap_norm, gt_tensor, threshold):
    Res_1 = (heatmap_norm > threshold).float()
    Res_0 = (heatmap_norm <= threshold).float()
    output = torch.stack([Res_0, Res_1], dim=0)
    output_AP = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)

    correct, labeled = batch_pix_accuracy(output, gt_tensor)
    inter, union = batch_intersection_union(output, gt_tensor, nclass=2)
    ap = get_ap_scores(output_AP, gt_tensor)
    ap_val = ap[0] if ap else 0.0

    return inter, union, correct, labeled, ap_val


def get_threshold(method, heatmap_norm, sparse_threshold, use_omp):
    """LeGrad always uses sparse_threshold. Others: mean (baseline) or sparse_threshold (OMP).
    Matches optimize_anti_hallucination.py / optimize_coco_anti_hallucination.py threshold logic."""
    if method == 'legrad':
        return sparse_threshold
    if use_omp:
        return sparse_threshold
    return heatmap_norm.mean().item()


def aggregate_metrics(results_dict):
    inter = results_dict['inter'].astype(np.float64)
    union = results_dict['union'].astype(np.float64)
    iou = inter / (union + 1e-10)
    miou = 100.0 * iou.mean()

    pix_acc = 100.0 * results_dict['pixel_correct'] / (results_dict['pixel_label'] + 1e-10)

    map_score = np.mean(results_dict['ap']) * 100 if results_dict['ap'] else 0.0

    return miou, pix_acc, map_score


def new_results_dict():
    return {
        'inter': np.zeros(2), 'union': np.zeros(2),
        'pixel_correct': 0, 'pixel_label': 0,
        'ap': [],
        'auroc_labels': [],
        'auroc_mean': [],
        'auroc_max': [],
        'auroc_energy': [],
    }


# ── Main evaluation ───────────────────────────────────────────────────────

def evaluate_method(config_name, config, coco_dir, metadata, image_size,
                    limit, seed, gemini_dict_path, gpt_dict_path, device):
    """Evaluate a single method on COCO: both baseline (no OMP) and with OMP."""
    method = config['method']
    strategy = config['dict_strategy']
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}  (dict={STRATEGY_NAMES[strategy]})")
    print(f"  Method: {method}, Model: {config['model_type']}")
    print(f"  OMP params: tau_act={config['sparse_threshold']}, T={config['atoms']}, "
          f"tau_cos={config['max_dict_cos_sim']}")
    if 'omp_beta' in config:
        print(f"  omp_beta={config['omp_beta']}")
    print(f"{'='*60}")

    gemini_dict = {}
    gpt_dict = {}
    if strategy == 'S3':
        path = gpt_dict_path or os.path.join(scripts_dir, 'visual_concept_dictionary_coco_gpt.json')
        gpt_dict = load_json_dictionary(path)
    elif strategy == 'S4':
        path = gemini_dict_path or os.path.join(scripts_dir, 'visual_concept_dictionary_coco.json')
        gemini_dict = load_json_dictionary(path)

    # Load model
    model = None
    tokenizer = None
    preprocess = None
    daam_segmenter = None

    if method == 'daam':
        if DAAMSegmenter is None:
            print(f"  SKIP: DAAMSegmenter not available")
            return None
        print(f"  Loading DAAMSegmenter ({config['daam_model_id']})...")
        daam_segmenter = DAAMSegmenter(model_id=config['daam_model_id'], device=device)
        print(f"  Loading CLIP ViT-B-16 for text embeddings...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', device=device
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        model = LeWrapper(model, layer_index=-2)
    else:
        print(f"  Loading {config['model_type']} model: {config['model_name']}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=config['model_name'], pretrained=config['pretrained'], device=device
        )
        tokenizer = open_clip.get_tokenizer(config['model_name'])
        model = LeWrapper(model, layer_index=-2)

    preprocess = LePreprocess(preprocess, image_size=image_size)

    image_dir = os.path.join(coco_dir, 'val2017')
    mask_dir = os.path.join(coco_dir, 'val2017_2_objects_masks')

    entries = list(metadata.values())
    if limit > 0:
        entries = entries[:limit]

    # Build text embeddings for all unique objects
    all_objects = set()
    for e in entries:
        all_objects.update(e['objects'])
    unique_objects = sorted(all_objects)
    obj_to_idx = {o: i for i, o in enumerate(unique_objects)}

    prompts = [f"a photo of a {obj}." for obj in unique_objects]
    tok_all = tokenizer(prompts).to(device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)

    ctx = {
        'tokenizer': tokenizer, 'model': model, 'device': device,
        'all_text_embs': all_text_embs, 'unique_objects': unique_objects,
        'gemini_dict': gemini_dict, 'gpt_dict': gpt_dict,
    }

    base_correct = new_results_dict()
    base_wrong = new_results_dict()
    omp_correct = new_results_dict()
    omp_wrong = new_results_dict()

    for entry in tqdm(entries, desc=f"  {config_name}"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            file_name = entry['file_name']
            objects = entry['objects']
            chosen = entry['chosen']
            other = [o for o in objects if o != chosen][0]

            img_path = os.path.join(image_dir, file_name)
            if not os.path.exists(img_path):
                continue
            base_img = Image.open(img_path).convert('RGB')
            img_t = preprocess(base_img).unsqueeze(0).to(device)

            mask_name = f"{file_name.replace('.jpg', '')}_{chosen.replace(' ', '_')}.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            gt_mask = load_mask(mask_path, (image_size, image_size))
            H_gt, W_gt = gt_mask.shape
            gt_tensor = torch.from_numpy(gt_mask).long()

            chosen_idx = obj_to_idx[chosen]
            other_idx = obj_to_idx[other]
            chosen_emb = all_text_embs[chosen_idx:chosen_idx + 1]
            other_emb = all_text_embs[other_idx:other_idx + 1]

            def process_heatmap(heatmap_raw, results_dict, auroc_label, threshold):
                hm_norm = normalize_heatmap(heatmap_raw, method, H_gt, W_gt)
                inter, union, correct, labeled, ap_val = compute_seg_metrics(hm_norm, gt_tensor, threshold)
                results_dict['inter'] += inter
                results_dict['union'] += union
                results_dict['pixel_correct'] += correct
                results_dict['pixel_label'] += labeled
                results_dict['ap'].append(ap_val)
                results_dict['auroc_labels'].append(auroc_label)
                H_hm, W_hm = heatmap_raw.shape[-2], heatmap_raw.shape[-1]
                hm_resized = F.interpolate(
                    heatmap_raw.view(1, 1, H_hm, W_hm), size=(H_gt, W_gt),
                    mode='bilinear', align_corners=False
                ).squeeze()
                hm_raw_np = hm_resized.detach().cpu().numpy() if isinstance(hm_resized, torch.Tensor) else hm_resized
                results_dict['auroc_mean'].append(float(np.mean(hm_raw_np)))
                results_dict['auroc_max'].append(float(np.max(hm_raw_np)))
                results_dict['auroc_energy'].append(float(np.sum(hm_raw_np[hm_raw_np > 0])))

            # ═══ BASELINE (no OMP) ═══

            # Correct prompt (chosen)
            if method == 'daam':
                hm_base_c = get_raw_heatmap(method, model, img_t, None,
                                            daam_segmenter=daam_segmenter, base_img=base_img,
                                            target_class_name=chosen, competing_concepts=[])
            else:
                hm_base_c = get_raw_heatmap(method, model, img_t, chosen_emb)

            hm_base_c_norm = normalize_heatmap(hm_base_c, method, H_gt, W_gt)
            thr_base_c = get_threshold(method, hm_base_c_norm, config['sparse_threshold'], use_omp=False)
            process_heatmap(hm_base_c, base_correct, auroc_label=1, threshold=thr_base_c)

            # Wrong prompt (other object in the same image)
            if method == 'daam':
                hm_base_w = get_raw_heatmap(method, model, img_t, None,
                                            daam_segmenter=daam_segmenter, base_img=base_img,
                                            target_class_name=other, competing_concepts=[])
            else:
                hm_base_w = get_raw_heatmap(method, model, img_t, other_emb)

            hm_base_w_norm = normalize_heatmap(hm_base_w, method, H_gt, W_gt)
            thr_base_w = get_threshold(method, hm_base_w_norm, config['sparse_threshold'], use_omp=False)
            process_heatmap(hm_base_w, base_wrong, auroc_label=0, threshold=thr_base_w)

            # ═══ OMP (with dictionary strategy) ═══

            ctx['target_idx'] = chosen_idx

            # Correct prompt — OMP
            if method == 'daam':
                competing = build_daam_competing_concepts(chosen, config, ctx)
                hm_omp_c = get_raw_heatmap(method, model, img_t, None,
                                           daam_segmenter=daam_segmenter, base_img=base_img,
                                           target_class_name=chosen, competing_concepts=competing,
                                           omp_beta=config.get('omp_beta', 1.0))
            else:
                sparse_c = build_sparse_embedding(chosen_emb, chosen, config, ctx)
                hm_omp_c = get_raw_heatmap(method, model, img_t, sparse_c)

            hm_omp_c_norm = normalize_heatmap(hm_omp_c, method, H_gt, W_gt)
            thr_omp_c = get_threshold(method, hm_omp_c_norm, config['sparse_threshold'], use_omp=True)
            process_heatmap(hm_omp_c, omp_correct, auroc_label=1, threshold=thr_omp_c)

            # Wrong prompt — OMP
            if method == 'daam':
                wrong_competing = [chosen]
                hm_omp_w = get_raw_heatmap(method, model, img_t, None,
                                           daam_segmenter=daam_segmenter, base_img=base_img,
                                           target_class_name=other,
                                           competing_concepts=wrong_competing,
                                           omp_beta=config.get('omp_beta', 1.0))
            else:
                ctx['target_idx'] = other_idx
                sparse_w = build_sparse_embedding(other_emb, other, config, ctx)
                hm_omp_w = get_raw_heatmap(method, model, img_t, sparse_w)

            hm_omp_w_norm = normalize_heatmap(hm_omp_w, method, H_gt, W_gt)
            thr_omp_w = get_threshold(method, hm_omp_w_norm, config['sparse_threshold'], use_omp=True)
            process_heatmap(hm_omp_w, omp_wrong, auroc_label=0, threshold=thr_omp_w)

        except Exception as e:
            print(f"  Error on {entry.get('file_name', '?')}: {e}")
            continue

    # Aggregate
    base_c_miou, base_c_acc, base_c_map = aggregate_metrics(base_correct)
    base_w_miou, base_w_acc, base_w_map = aggregate_metrics(base_wrong)

    omp_c_miou, omp_c_acc, omp_c_map = aggregate_metrics(omp_correct)
    omp_w_miou, omp_w_acc, omp_w_map = aggregate_metrics(omp_wrong)

    def compute_auroc_variants(correct_dict, wrong_dict):
        labels = correct_dict['auroc_labels'] + wrong_dict['auroc_labels']
        if len(labels) < 2 or len(set(labels)) < 2:
            return {'mean': float('nan'), 'max': float('nan'), 'energy': float('nan')}
        out = {}
        for key in ('mean', 'max', 'energy'):
            scores = correct_dict[f'auroc_{key}'] + wrong_dict[f'auroc_{key}']
            out[key] = roc_auc_score(labels, scores) * 100.0
        return out

    base_aurocs = compute_auroc_variants(base_correct, base_wrong)
    omp_aurocs = compute_auroc_variants(omp_correct, omp_wrong)

    print(f"\n  {'─'*72}")
    print(f"  Results for {config_name}:")
    print(f"  {'─'*72}")
    print(f"  {'':>20} {'mIoU':>8} {'Acc':>8} {'mAP':>8} │ {'AUC(mean)':>10} {'AUC(max)':>10} {'AUC(energy)':>12}")
    print(f"  {'─'*72}")
    print(f"  {'Baseline correct':>20} {base_c_miou:>7.2f}% {base_c_acc:>7.2f}% {base_c_map:>7.2f}% │")
    print(f"  {'Baseline wrong':>20} {base_w_miou:>7.2f}% {base_w_acc:>7.2f}% {base_w_map:>7.2f}% │")
    print(f"  {'Baseline AUROC':>20} {'':>8} {'':>8} {'':>8} │ "
          f"{base_aurocs['mean']:>9.2f}% {base_aurocs['max']:>9.2f}% {base_aurocs['energy']:>11.2f}%")
    print(f"  {'─'*72}")
    print(f"  {'OMP correct':>20} {omp_c_miou:>7.2f}% {omp_c_acc:>7.2f}% {omp_c_map:>7.2f}% │")
    print(f"  {'OMP wrong':>20} {omp_w_miou:>7.2f}% {omp_w_acc:>7.2f}% {omp_w_map:>7.2f}% │")
    print(f"  {'OMP AUROC':>20} {'':>8} {'':>8} {'':>8} │ "
          f"{omp_aurocs['mean']:>9.2f}% {omp_aurocs['max']:>9.2f}% {omp_aurocs['energy']:>11.2f}%")
    print(f"  {'─'*72}")

    return {
        'config_name': config_name,
        'method': method,
        'model_type': config['model_type'],
        'strategy': config['dict_strategy'],
        'baseline': {
            'correct': {'miou': base_c_miou, 'acc': base_c_acc, 'map': base_c_map},
            'wrong': {'miou': base_w_miou, 'acc': base_w_acc, 'map': base_w_map},
            'auroc_mean': base_aurocs['mean'],
            'auroc_max': base_aurocs['max'],
            'auroc_energy': base_aurocs['energy'],
        },
        'omp': {
            'correct': {'miou': omp_c_miou, 'acc': omp_c_acc, 'map': omp_c_map},
            'wrong': {'miou': omp_w_miou, 'acc': omp_w_acc, 'map': omp_w_map},
            'auroc_mean': omp_aurocs['mean'],
            'auroc_max': omp_aurocs['max'],
            'auroc_energy': omp_aurocs['energy'],
        },
        'hyperparameters': {
            'sparse_threshold': config['sparse_threshold'],
            'atoms': config['atoms'],
            'max_dict_cos_sim': config['max_dict_cos_sim'],
            'omp_beta': config.get('omp_beta', None),
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute image-level AUROC + segmentation metrics on MS COCO 2017',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/compute_image_level_auroc_coco.py --limit 100
  python scripts/compute_image_level_auroc_coco.py --strategies S3 --limit 100
  python scripts/compute_image_level_auroc_coco.py --strategies S1 S4 --methods legrad_clip gradcam_siglip
""")
    default_coco = os.path.join(project_root, 'ms_coco_2017')
    parser.add_argument('--coco_dir', type=str, default=default_coco,
                        help='Root directory for MS COCO 2017 data')
    parser.add_argument('--metadata_json', type=str, default=None,
                        help='Path to selected_images_2_unique.json (default: <coco_dir>/selected_images_2_unique.json)')
    parser.add_argument('--gemini_dict_path', type=str, default=None,
                        help='Path to Gemini/LLM COCO dictionary (default: scripts/visual_concept_dictionary_coco.json)')
    parser.add_argument('--gpt_dict_path', type=str, default=None,
                        help='Path to GPT COCO dictionary (default: scripts/visual_concept_dictionary_coco_gpt.json)')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--limit', type=int, default=0, help='Max images (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default=None, help='Save results JSON to this path')
    parser.add_argument('--strategies', nargs='+', default=None,
                        choices=DICT_STRATEGIES,
                        help='Which dictionary strategies to run (default: all S1-S4)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Which method_model pairs to run (e.g. legrad_clip gradcam_siglip daam_daam)')
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    if args.metadata_json is None:
        args.metadata_json = os.path.join(args.coco_dir, 'selected_images_2_unique.json')

    if not os.path.exists(args.metadata_json):
        print(f"ERROR: Metadata file not found: {args.metadata_json}")
        print(f"  Expected at: <coco_dir>/selected_images_2_unique.json")
        return

    with open(args.metadata_json, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} image entries from metadata")

    strategies = args.strategies if args.strategies else DICT_STRATEGIES
    configs = build_all_configs(strategies=strategies, methods=args.methods)

    if not configs:
        valid = [f"{m}_{mk}" for m, mk in METHOD_MODEL_PAIRS]
        print(f"No configs matched. Valid --methods values: {valid}")
        return

    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size}")
    print(f"Limit: {args.limit if args.limit > 0 else 'all'}")
    print(f"Seed: {args.seed}")
    print(f"Strategies: {strategies}")
    print(f"Configs to run: {len(configs)}")

    results = {}
    for config_name, config in configs.items():
        result = evaluate_method(
            config_name, config,
            coco_dir=args.coco_dir,
            metadata=metadata,
            image_size=args.image_size,
            limit=args.limit,
            seed=args.seed,
            gemini_dict_path=args.gemini_dict_path,
            gpt_dict_path=args.gpt_dict_path,
            device=args.device,
        )
        if result is not None:
            results[config_name] = result

    # ── Summary comparison table ──
    W = 115
    print(f"\n\n{'='*W}")
    print("SUMMARY: BASELINE vs OMP — MS COCO 2017 (all dictionary strategies)")
    print(f"{'='*W}")

    header = (f"{'Config':<30} │ {'mIoU(C)':>8} {'Acc(C)':>8} {'mAP(C)':>8} │ "
              f"{'mIoU(W)':>8} {'Acc(W)':>8} {'mAP(W)':>8} │ "
              f"{'AUC(mean)':>10} {'AUC(max)':>10} {'AUC(nrg)':>10}")
    print(header)
    print(f"{'─'*W}")

    sign_fn = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"

    for name, r in results.items():
        b = r['baseline']
        o = r['omp']
        print(f"{name + ' (base)':<30} │ "
              f"{b['correct']['miou']:>7.2f}% {b['correct']['acc']:>7.2f}% {b['correct']['map']:>7.2f}% │ "
              f"{b['wrong']['miou']:>7.2f}% {b['wrong']['acc']:>7.2f}% {b['wrong']['map']:>7.2f}% │ "
              f"{b['auroc_mean']:>9.2f}% {b['auroc_max']:>9.2f}% {b['auroc_energy']:>9.2f}%")
        print(f"{name + ' (OMP)':<30} │ "
              f"{o['correct']['miou']:>7.2f}% {o['correct']['acc']:>7.2f}% {o['correct']['map']:>7.2f}% │ "
              f"{o['wrong']['miou']:>7.2f}% {o['wrong']['acc']:>7.2f}% {o['wrong']['map']:>7.2f}% │ "
              f"{o['auroc_mean']:>9.2f}% {o['auroc_max']:>9.2f}% {o['auroc_energy']:>9.2f}%")

        d_c_miou = o['correct']['miou'] - b['correct']['miou']
        d_w_miou = o['wrong']['miou'] - b['wrong']['miou']
        d_auc_mean = o['auroc_mean'] - b['auroc_mean']
        d_auc_max = o['auroc_max'] - b['auroc_max']
        d_auc_nrg = o['auroc_energy'] - b['auroc_energy']
        print(f"{'  Δ':<30} │ "
              f"{sign_fn(d_c_miou) + '%':>8} {'':>8} {'':>8} │ "
              f"{sign_fn(d_w_miou) + '%':>8} {'':>8} {'':>8} │ "
              f"{sign_fn(d_auc_mean) + '%':>10} {sign_fn(d_auc_max) + '%':>10} {sign_fn(d_auc_nrg) + '%':>10}")
        print(f"{'─'*W}")

    print(f"{'='*W}")
    print("\nS1=Classes  S2=WordNet  S3=GPT-OSS  S4=Gemini")
    print("AUC(mean)=mean heatmap  AUC(max)=peak heatmap  AUC(nrg)=sum above threshold")

    if args.output:
        with open(args.output, 'w') as out_f:
            json.dump(results, out_f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
