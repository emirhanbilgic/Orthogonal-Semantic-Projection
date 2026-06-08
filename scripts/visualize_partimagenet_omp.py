#!/usr/bin/env python3
"""
Visualize LeGrad + OMP on PartImageNet subset.

For a few images, generates a grid showing:
1. Original image
2. For each part in the image: 
   - Ground truth mask
   - Predicted raw heatmap
   - Thresholded heatmap
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import open_clip
from legrad import LeWrapper, LePreprocess
from sparse_encoding import omp_sparse_residual, compute_map_for_embedding

def get_image_parts(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    category_ids = list(set([ann['category_id'] for ann in anns]))
    categories = coco.loadCats(category_ids)
    return categories

def get_binary_mask(coco, img_id, cat_id, target_size):
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if len(anns) == 0:
        return np.zeros(target_size, dtype=np.uint8)
    mask = np.zeros((coco.imgs[img_id]['height'], coco.imgs[img_id]['width']), dtype=np.uint8)
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))
    mask_img = Image.fromarray(mask * 255)
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    arr = np.array(mask_img)
    return (arr > 128).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Visualize LeGrad+OMP on PartImageNet')
    parser.add_argument('--dataset_dir', type=str, default='partimagenet_1000_subset')
    parser.add_argument('--limit', type=int, default=3, help='Number of images to visualize')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--atoms', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'outputs', 'partimagenet_omp'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    annotations_file = os.path.join(args.dataset_dir, 'subset_annotations.json')
    images_dir = os.path.join(args.dataset_dir, 'images')
    coco = COCO(annotations_file)
    img_ids = coco.getImgIds()[:args.limit]

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        base_img = Image.open(img_path).convert('RGB')
        img_t = preprocess(base_img).unsqueeze(0).to(args.device)
        
        categories = get_image_parts(coco, img_id)
        if len(categories) < 2:
            continue
            
        cat_names = [cat['name'] for cat in categories]
        prompts = [f"a photo of a {name}." for name in cat_names]
        
        tok = tokenizer(prompts).to(args.device)
        with torch.no_grad():
            text_embs = model.encode_text(tok, normalize=True)

        n_parts = len(categories)
        fig, axes = plt.subplots(n_parts, 4, figsize=(16, 4 * n_parts))
        fig.suptitle(f"Image ID: {img_id}\nParts in Image: {', '.join(cat_names)}", fontsize=16)

        for i, target_cat in enumerate(categories):
            target_emb = text_embs[i:i+1]
            dictionary_embs = torch.cat([text_embs[:i], text_embs[i+1:]], dim=0)
            
            # OMP sparse text embedding
            sparse_emb = omp_sparse_residual(target_emb, dictionary_embs, max_atoms=args.atoms)
            
            # Heatmap
            heatmap = compute_map_for_embedding(model, img_t, sparse_emb)
            heatmap_np = heatmap.squeeze().detach().cpu().numpy()
            
            # GT Mask
            gt_mask = get_binary_mask(coco, img_id, target_cat['id'], (args.image_size, args.image_size))
            
            # Resize heatmap to match image size
            heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)
            heatmap_resized = F.interpolate(
                heatmap_tensor, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False
            ).squeeze().numpy()
            
            legrad_bin = (heatmap_resized > args.threshold).astype(np.uint8)
            vis_img = base_img.resize((args.image_size, args.image_size))
            
            ax_img = axes[i, 0] if n_parts > 1 else axes[0]
            ax_gt = axes[i, 1] if n_parts > 1 else axes[1]
            ax_hm = axes[i, 2] if n_parts > 1 else axes[2]
            ax_bin = axes[i, 3] if n_parts > 1 else axes[3]
            
            ax_img.imshow(vis_img)
            ax_img.set_title(f"Image")
            ax_img.axis("off")
            
            ax_gt.imshow(gt_mask, cmap="gray")
            ax_gt.set_title(f"GT: {target_cat['name']}")
            ax_gt.axis("off")
            
            ax_hm.imshow(heatmap_resized, cmap="viridis")
            ax_hm.set_title(f"Predicted Heatmap")
            ax_hm.axis("off")
            
            ax_bin.imshow(legrad_bin, cmap="gray")
            ax_bin.set_title(f"OMP Threshold ({args.threshold})")
            ax_bin.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(args.output_dir, f"partimagenet_omp_vis_{img_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
