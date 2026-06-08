#!/usr/bin/env python3
"""
Evaluate LeGrad + OMP on PartImageNet subset.

For each target part in an image, the script uses the other parts within the SAME image
as the negative dictionary for Orthogonal Matching Pursuit (OMP), then uses the text embedding
residual with LeGrad to compute heatmaps.
Calculates mIoU, Pixel Accuracy, mAP, and AUROC.
"""

import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from tqdm import tqdm
from pycocotools.coco import COCO

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
from benchmark_segmentation import batch_intersection_union, batch_pix_accuracy, get_ap_scores


def get_image_parts(coco, img_id):
    """Get all unique parts (categories) annotated in an image."""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    category_ids = list(set([ann['category_id'] for ann in anns]))
    categories = coco.loadCats(category_ids)
    return categories


def get_binary_mask(coco, img_id, cat_id, target_size):
    """Get binary mask for a specific category in an image."""
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


def compute_metrics(heatmap_np, gt_mask, thr):
    """Compute all metrics for a single prediction and GT pair."""
    # Resize GT mask shape
    H_gt, W_gt = gt_mask.shape
    H_hm, W_hm = heatmap_np.shape
    
    heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(H_gt, W_gt),
        mode='bilinear',
        align_corners=False,
    ).squeeze()
    
    # Thresholding
    Res_1 = (heatmap_resized > thr).float()
    Res_0 = (heatmap_resized <= thr).float()
    output_tensor = torch.stack([Res_0, Res_1], dim=0)       # [2,H,W]
    output_AP     = torch.stack([1.0 - heatmap_resized, heatmap_resized], dim=0)

    gt_tensor = torch.from_numpy(gt_mask).long()

    inter, union = batch_intersection_union(output_tensor, gt_tensor, nclass=2)
    ap_list = get_ap_scores(output_AP, gt_tensor)
    ap = ap_list[0] if ap_list else 0.0

    correct_pixels, labeled_pixels = batch_pix_accuracy(output_tensor, gt_tensor)

    heatmap_flat = heatmap_resized.detach().cpu().numpy().flatten()
    gt_flat = gt_mask.flatten()
    
    return inter, union, correct_pixels, labeled_pixels, ap, heatmap_flat, gt_flat


def main():
    parser = argparse.ArgumentParser(description='Evaluate LeGrad+OMP on PartImageNet')
    parser.add_argument('--dataset_dir', type=str, default='partimagenet_1000_subset')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to process')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--sparse_threshold', type=float, default=0.85, help='Threshold for IoU and Accuracy')
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.65)
    parser.add_argument('--max_atoms', type=int, default=3, help='Maximum number of atoms for OMP')
    parser.add_argument('--output', type=str, default='partimagenet_omp_results_custom.json')
    args = parser.parse_args()

    annotations_file = os.path.join(args.dataset_dir, 'subset_annotations.json')
    images_dir = os.path.join(args.dataset_dir, 'images')

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

    print(f"Loading annotations from {annotations_file}")
    coco = COCO(annotations_file)
    img_ids = coco.getImgIds()
    
    if args.limit > 0:
        img_ids = img_ids[:args.limit]
        
    print(f"Processing {len(img_ids)} images")

    # Metrics accumulators
    eval_methods = ['baseline', 'omp']
    accumulators = {
        method: {
            'total_inter': np.zeros(2),
            'total_union': np.zeros(2),
            'total_correct': 0,
            'total_labeled': 0,
            'all_aps': []
        }
        for method in eval_methods
    }

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        try:
            base_img = Image.open(img_path).convert('RGB')
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            
            categories = get_image_parts(coco, img_id)
            if len(categories) < 2:
                continue # Need at least 2 parts for dictionary
                
            cat_names = [cat['name'] for cat in categories]
            prompts = [f"a photo of a {name}." for name in cat_names]
            
            tok = tokenizer(prompts).to(args.device)
            with torch.no_grad():
                text_embs = model.encode_text(tok, normalize=True)
                
            for i, target_cat in enumerate(categories):
                target_emb = text_embs[i:i+1]
                
                # Dictionary is everything ELSE in the image
                dictionary_embs = torch.cat([text_embs[:i], text_embs[i+1:]], dim=0)
                
                # Filter by max dict cosine similarity
                if dictionary_embs.shape[0] > 0 and 0.0 < args.max_dict_cos_sim < 1.0:
                    sim = (dictionary_embs @ target_emb.t()).squeeze(-1).abs()
                    keep = sim < args.max_dict_cos_sim
                    dictionary_embs = dictionary_embs[keep]
                
                # Cap atoms via parameter
                num_negativas = dictionary_embs.shape[0]
                atoms = min(args.max_atoms, num_negativas)
                
                # OMP Sparse encoding
                sparse_emb = omp_sparse_residual(target_emb, dictionary_embs, max_atoms=atoms)
                
                # Compute Heatmaps
                heatmaps = {
                    'baseline': compute_map_for_embedding(model, img_t, target_emb).squeeze().detach().cpu().numpy(),
                    'omp': compute_map_for_embedding(model, img_t, sparse_emb).squeeze().detach().cpu().numpy()
                }
                
                # GT Mask
                gt_mask = get_binary_mask(coco, img_id, target_cat['id'], (args.image_size, args.image_size))
                
                # Skip if empty mask
                if gt_mask.sum() == 0:
                   continue
                
                for method in eval_methods:
                    # Metrics
                    current_threshold = 0.5 if method == 'baseline' else args.sparse_threshold
                    inter, union, c_p, l_p, ap, hm_flat, gt_flat = compute_metrics(heatmaps[method], gt_mask, current_threshold)
                    
                    accumulators[method]['total_inter'] += inter
                    accumulators[method]['total_union'] += union
                    accumulators[method]['total_correct'] += c_p
                    accumulators[method]['total_labeled'] += l_p
                    accumulators[method]['all_aps'].append(ap)

        except Exception as e:
            print(f"Error on image {img_id}: {e}")
            continue

    final_results = {}
    for method in eval_methods:
        accums = accumulators[method]
        iou = accums['total_inter'].astype(np.float64) / (accums['total_union'].astype(np.float64) + 1e-10)
        miou = 100.0 * iou.mean()
        acc = 100.0 * accums['total_correct'] / (accums['total_labeled'] + 1e-10)
        map_score = np.mean(accums['all_aps']) * 100 if accums['all_aps'] else 0.0
        
        method_threshold = 0.5 if method == 'baseline' else args.sparse_threshold
        print(f"\nResults for {method.upper()} (Threshold {method_threshold}, Max Dict Cos Sim: {args.max_dict_cos_sim})")
        print(f"mIoU: {miou:.2f}")
        print(f"Pixel Accuracy: {acc:.2f}")
        print(f"mAP: {map_score:.2f}")
        print(f"Number of target parts evaluated: {len(accums['all_aps'])}")
        
        final_results[method] = {
            'miou': float(miou),
            'pixel_accuracy': float(acc),
            'map': float(map_score),
            'n_samples': len(accums['all_aps'])
        }

    results = {
        'settings': {
            'dataset': args.dataset_dir,
            'limit': args.limit,
            'model': args.model_name,
            'pretrained': args.pretrained,
            'sparse_threshold': args.sparse_threshold,
            'max_dict_cos_sim': args.max_dict_cos_sim,
            'max_atoms': args.max_atoms
        },
        'metrics': final_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
