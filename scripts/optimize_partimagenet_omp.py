#!/usr/bin/env python3
"""
Optimize LeGrad + OMP hyperparameters on PartImageNet subset using Optuna.

Tunes:
- max_atoms (integer: 1 to number of negative classes)
- threshold (float: 0.1 to 0.9)

Objective: Maximize mIoU on a subset of images (e.g., 50).
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
from pycocotools.coco import COCO
import optuna

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

def compute_metrics(heatmap_np, gt_mask, sparse_threshold):
    H_gt, W_gt = gt_mask.shape
    heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor, size=(H_gt, W_gt), mode='bilinear', align_corners=False,
    ).squeeze()
    
    Res_1 = (heatmap_resized > sparse_threshold).float()
    Res_0 = (heatmap_resized <= sparse_threshold).float()
    output_tensor = torch.stack([Res_0, Res_1], dim=0)
    output_AP = torch.stack([1.0 - heatmap_resized, heatmap_resized], dim=0)
    gt_tensor = torch.from_numpy(gt_mask).long()
    
    inter, union = batch_intersection_union(output_tensor, gt_tensor, nclass=2)
    c_p, l_p = batch_pix_accuracy(output_tensor, gt_tensor)
    
    ap_list = get_ap_scores(output_AP, gt_tensor)
    ap = ap_list[0] if ap_list else 0.0
        
    return inter, union, c_p, l_p, ap


def evaluate_config(coco, images_dir, img_ids, model, tokenizer, preprocess, device, max_atoms_cap, sparse_threshold, max_dict_cos_sim, image_size):
    total_inter = np.zeros(2)
    total_union = np.zeros(2)
    total_correct = 0
    total_labeled = 0
    all_aps = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        try:
            base_img = Image.open(img_path).convert('RGB')
            img_t = preprocess(base_img).unsqueeze(0).to(device)
            
            categories = get_image_parts(coco, img_id)
            if len(categories) < 2:
                continue 
                
            cat_names = [cat['name'] for cat in categories]
            prompts = [f"a photo of a {name}." for name in cat_names]
            tok = tokenizer(prompts).to(device)
            with torch.no_grad():
                text_embs = model.encode_text(tok, normalize=True)
                
            for i, target_cat in enumerate(categories):
                target_emb = text_embs[i:i+1]
                dictionary_embs = torch.cat([text_embs[:i], text_embs[i+1:]], dim=0)
                
                # Filter dictionary by cosine similarity
                if dictionary_embs.shape[0] > 0 and 0.0 < max_dict_cos_sim < 1.0:
                    sim = (dictionary_embs @ target_emb.t()).squeeze(-1).abs()
                    keep = sim < max_dict_cos_sim
                    dictionary_embs = dictionary_embs[keep]
                
                # Dynamic atoms cap based on dictionary size or parameter
                num_negativas = dictionary_embs.shape[0]
                atoms = min(max_atoms_cap, num_negativas)
                
                sparse_emb = omp_sparse_residual(target_emb, dictionary_embs, max_atoms=atoms)
                heatmap = compute_map_for_embedding(model, img_t, sparse_emb)
                heatmap_np = heatmap.squeeze().detach().cpu().numpy()
                
                gt_mask = get_binary_mask(coco, img_id, target_cat['id'], (image_size, image_size))
                if gt_mask.sum() == 0:
                   continue
                
                inter, union, c_p, l_p, ap = compute_metrics(heatmap_np, gt_mask, sparse_threshold)
                total_inter += inter
                total_union += union
                total_correct += c_p
                total_labeled += l_p
                all_aps.append(ap)
                
        except Exception as e:
            print(f"Error on image {img_id}: {e}")
            continue
            
    iou = total_inter.astype(np.float64) / (total_union.astype(np.float64) + 1e-10)
    miou = 100.0 * iou.mean()
    acc = 100.0 * total_correct / (total_labeled + 1e-10)
    map_score = np.mean(all_aps) * 100 if all_aps else 0.0
    
    return miou, acc, map_score


def main():
    parser = argparse.ArgumentParser(description='Optimize LeGrad+OMP hyperparameters with Optuna')
    parser.add_argument('--dataset_dir', type=str, default='partimagenet_1000_subset')
    parser.add_argument('--limit', type=int, default=50, help='Number of images to evaluate per trial')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    annotations_file = os.path.join(args.dataset_dir, 'subset_annotations.json')
    images_dir = os.path.join(args.dataset_dir, 'images')

    print(f"Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    coco = COCO(annotations_file)
    all_img_ids = coco.getImgIds()
    
    # We fix the subset of images evaluated across all trials so comparisons are apples-to-apples.
    np.random.seed(42)
    img_ids_subset = np.random.choice(all_img_ids, size=min(args.limit, len(all_img_ids)), replace=False).tolist()
    print(f"Loaded dataset. Optimizing on fixed subset of {len(img_ids_subset)} images.")

    def objective(trial):
        # Hyperparameters
        sparse_threshold = trial.suggest_float('sparse_threshold', 0.1, 0.9, step=0.025)
        max_dict_cos_sim = trial.suggest_float('max_dict_cos_sim', 0.5, 1.0, step=0.05)
        max_atoms_cap = trial.suggest_int('max_atoms', 1, 5)
        
        miou, acc, map_score = evaluate_config(
            coco=coco,
            images_dir=images_dir,
            img_ids=img_ids_subset,
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=args.device,
            max_atoms_cap=max_atoms_cap,
            sparse_threshold=sparse_threshold,
            max_dict_cos_sim=max_dict_cos_sim,
            image_size=args.image_size
        )
        
        trial.set_user_attr('miou', miou)
        trial.set_user_attr('acc', acc)
        trial.set_user_attr('map', map_score)
        
        combined_score = (miou + acc + map_score) / 3.0
        
        print(f"mIoU: {miou:.2f}")
        print(f"Pixel Accuracy: {acc:.2f}")
        print(f"mAP: {map_score:.2f}")
        
        return combined_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- Optuna Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Score): {trial.value:.2f}%")
    print(f"  mIoU: {trial.user_attrs.get('miou', 0.0):.2f}%")
    print(f"  Pixel Accuracy: {trial.user_attrs.get('acc', 0.0):.2f}%")
    print(f"  mAP: {trial.user_attrs.get('map', 0.0):.2f}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save optimized parameters
    output = {
        'best_avg_score': float(trial.value),
        'best_miou': float(trial.user_attrs.get('miou', 0.0)),
        'best_acc': float(trial.user_attrs.get('acc', 0.0)),
        'best_map': float(trial.user_attrs.get('map', 0.0)),
        'best_params': trial.params
    }
    out_file = "partimagenet_omp_optuna_best.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved optimized parameters to {out_file}")

if __name__ == '__main__':
    main()
