#!/usr/bin/env python3
"""
Dictionary-size ablation for ECCV 2026 rebuttal Table 3.

Sweeps over # atoms with a fixed dictionary strategy and reports paired
AUROC at each size. Auto-selects atom grid based on dictionary type:
  - Gemini/GPT dictionaries: {1, 3, 5, 10, 20, 30, 40} (max ~40 per class)
  - Class-based (S1/S2):     {1, 3, 5, 10, 20, 50, 100} (max ~444+)

Uses the same evaluation infrastructure as optimize_anti_hallucination.py
(AntiHallucinationObjective.evaluate_sparse_config) so results are directly
comparable to the main paper.

Usage:
  # Gemini dictionary (S3), LeGrad + CLIP, full dataset
  python3 scripts/dictionary_size_ablation.py \
      --use_llm_dictionary --fix_dictionary --limit 4276

  # WordNet dictionary (S2)
  python3 scripts/dictionary_size_ablation.py \
      --fix_dictionary --limit 4276

  # Quick test (100 images)
  python3 scripts/dictionary_size_ablation.py \
      --use_llm_dictionary --fix_dictionary --limit 100
"""

import sys, os, json, argparse, time
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import open_clip
from legrad import LeWrapper, LePreprocess

from benchmark_segmentation import (
    load_imagenet_class_index, build_wnid_to_label_map,
)
from optimize_anti_hallucination import AntiHallucinationObjective, BASELINES

# Default atom counts to sweep for the dictionary-size ablation.
# The expanded dictionary (visual_concept_dictionary_445_80.json) has
# ~58-116 concepts per class, so atoms up to 80 is supported.
# For S1/S2 (class-based), the pool is ~444+ so larger values work fine.
DEFAULT_ATOM_GRID_LLM = [3, 5, 10, 20, 40, 60, 80]
DEFAULT_ATOM_GRID_CLASSES = [3, 5, 10, 20, 50, 100, 200]  # for S1/S2 (max ~444+)


def main():
    parser = argparse.ArgumentParser(
        description='Dictionary-size ablation (# atoms sweep)')
    parser.add_argument('--mat_file', default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_json', default='dictionary_ablation_results.json')

    # model
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--use_siglip', action='store_true')
    parser.add_argument('--class_index_path', default='resources/imagenet_class_index.json')

    # method (same flags as optimize_anti_hallucination.py)
    parser.add_argument('--use_gradcam', action='store_true')
    parser.add_argument('--use_chefercam', action='store_true')
    parser.add_argument('--use_attentioncam', action='store_true')
    parser.add_argument('--use_daam', action='store_true')
    parser.add_argument('--use_daam_keyspace_omp', action='store_true')
    parser.add_argument('--daam_model_id', default='Manojb/stable-diffusion-2-base')

    # dictionary config
    parser.add_argument('--fix_dictionary', action='store_true')
    parser.add_argument('--fix_dictionary_wordnet_only', action='store_true')
    parser.add_argument('--fix_dictionary_prompts_only', action='store_true')
    parser.add_argument('--use_llm_dictionary', action='store_true')
    parser.add_argument('--llm_dictionary_path', default=None)
    parser.add_argument('--use_gpt_dictionary', action='store_true')
    parser.add_argument('--gpt_dictionary_path', default=None)

    # ablation parameters
    parser.add_argument('--atom_grid', type=int, nargs='+', default=None,
                        help='List of atom counts to sweep (auto-selected if not given)')
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.6,
                        help='τ_cos threshold (default 0.6 = Gemini S3 LeGrad/CLIP)')
    parser.add_argument('--sparse_threshold', type=float, default=0.4,
                        help='Binarization threshold τ_act (default 0.4 = LeGrad/CLIP)')

    args = parser.parse_args()

    # Auto-select atom grid based on dictionary type
    if args.atom_grid is None:
        if args.use_llm_dictionary or args.use_gpt_dictionary:
            args.atom_grid = DEFAULT_ATOM_GRID_LLM
            print(f"[auto] Using LLM atom grid (max ~40): {args.atom_grid}")
        else:
            args.atom_grid = DEFAULT_ATOM_GRID_CLASSES
            print(f"[auto] Using class-based atom grid: {args.atom_grid}")

    # ── model setup ──────────────────────────────────────────────────
    if args.use_siglip:
        args.model_name = args.model_name or 'ViT-B-16-SigLIP'
        args.pretrained = args.pretrained or 'webli'
        model_label = 'SigLIP'
        model_type_key = 'SigLIP'
    else:
        args.model_name = args.model_name or 'ViT-B-16'
        args.pretrained = args.pretrained or 'laion2b_s34b_b88k'
        model_label = 'CLIP'
        model_type_key = 'CLIP'

    if args.use_gradcam:        method_label = 'GradCAM'
    elif args.use_chefercam:    method_label = 'CheferCAM'
    elif args.use_attentioncam: method_label = 'AttentionCAM'
    elif args.use_daam:         method_label = 'DAAM'
    else:                       method_label = 'LeGrad'

    if args.use_daam:
        model_type_key = 'DAAM'
        method_key = 'DAAM'
    else:
        method_key = method_label

    baseline_metrics = BASELINES[model_type_key][method_key]
    baseline_auroc = baseline_metrics['correct'].get('auroc', 0)

    print(f"{'='*70}")
    print(f"DICTIONARY-SIZE ABLATION")
    print(f"{'='*70}")
    print(f"Model:  {model_label} | Method: {method_label}")
    print(f"Atoms:  {args.atom_grid}")
    print(f"τ_cos:  {args.max_dict_cos_sim}  |  τ_act: {args.sparse_threshold}")
    print(f"Baseline AUROC: {baseline_auroc:.2f}")
    print(f"{'='*70}\n")

    device = args.device
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # class index
    try:
        ci = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(ci)
    except Exception:
        wnid_to_label = {}

    # NLTK
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception:
        pass

    # ── create objective (reuses full evaluation infrastructure) ──────
    objective = AntiHallucinationObjective(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        dataset_file=args.mat_file,
        wnid_to_label=wnid_to_label,
        device=device,
        image_size=args.image_size,
        limit=args.limit,
        num_negatives=1,
        negative_strategy='random',
        seed=args.seed,
        use_gradcam=args.use_gradcam,
        use_chefercam=args.use_chefercam,
        use_attentioncam=args.use_attentioncam,
        use_daam=args.use_daam,
        use_daam_keyspace_omp=args.use_daam_keyspace_omp,
        daam_model_id=args.daam_model_id,
        fix_dictionary=args.fix_dictionary,
        fix_dictionary_wordnet_only=args.fix_dictionary_wordnet_only,
        fix_dictionary_prompts_only=args.fix_dictionary_prompts_only,
        use_llm_dictionary=args.use_llm_dictionary,
        llm_dictionary_path=args.llm_dictionary_path,
        use_gpt_dictionary=args.use_gpt_dictionary,
        gpt_dictionary_path=args.gpt_dictionary_path,
        baseline_metrics=baseline_metrics,
        threshold_mode='fixed',
        fixed_threshold=args.sparse_threshold,
    )

    # Determine dictionary flags (same logic as AntiHallucinationObjective.__call__)
    if args.use_llm_dictionary or args.use_gpt_dictionary:
        wn_syn, wn_hyper, wn_hypo, wn_sib = False, False, False, False
        dict_prompts = False
    elif args.fix_dictionary:
        wn_syn, wn_hyper, wn_hypo, wn_sib = True, True, True, False
        dict_prompts = True
    elif args.fix_dictionary_wordnet_only:
        wn_syn, wn_hyper, wn_hypo, wn_sib = False, True, True, True
        dict_prompts = False
    elif args.fix_dictionary_prompts_only:
        wn_syn, wn_hyper, wn_hypo, wn_sib = False, False, False, False
        dict_prompts = True
    else:
        # Default: all WordNet + prompts
        wn_syn, wn_hyper, wn_hypo, wn_sib = True, True, True, True
        dict_prompts = True

    # ── sweep over atom counts ───────────────────────────────────────
    results = []

    for atoms in args.atom_grid:
        print(f"\n{'─'*50}")
        print(f"  Evaluating atoms={atoms}")
        print(f"{'─'*50}")

        t0 = time.time()

        (c_miou, w_miou, c_acc, w_acc, c_map, w_map,
         c_auroc, w_auroc, c_auroc_max, c_auroc_min, c_auroc_median,
         w_auroc_max, w_auroc_min, w_auroc_median,
         c_stats, w_stats) = objective.evaluate_sparse_config(
            wn_use_synonyms=wn_syn,
            wn_use_hypernyms=wn_hyper,
            wn_use_hyponyms=wn_hypo,
            wn_use_siblings=wn_sib,
            dict_include_prompts=dict_prompts,
            sparse_threshold=args.sparse_threshold,
            atoms=atoms,
            max_dict_cos_sim=args.max_dict_cos_sim,
            omp_beta=1.0,
            show_progress=True,
        )

        elapsed = time.time() - t0
        delta_auroc = c_auroc - baseline_auroc

        entry = {
            'atoms': atoms,
            'correct_auroc': round(c_auroc, 2),
            'delta_auroc': round(delta_auroc, 2),
            'correct_miou': round(c_miou, 2),
            'wrong_miou': round(w_miou, 2),
            'correct_map': round(c_map, 2),
            'correct_acc': round(c_acc, 2),
            'wrong_auroc': round(w_auroc, 2),
            'elapsed_sec': round(elapsed, 1),
        }
        results.append(entry)

        print(f"  atoms={atoms:3d} | AUROC={c_auroc:.2f} (Δ={delta_auroc:+.2f}) "
              f"| mIoU={c_miou:.2f} | mAP={c_map:.2f} | {elapsed:.0f}s")

    # ── summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DICTIONARY-SIZE ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"Model: {model_label} | Method: {method_label}")
    print(f"Baseline AUROC: {baseline_auroc:.2f}")
    print()

    # Print as a table
    header = "  # Atoms |  AUROC  |   Δ    |  mIoU  |  mAP"
    print(header)
    print("  " + "─" * len(header))
    for r in results:
        print(f"  {r['atoms']:7d} | {r['correct_auroc']:6.2f}% | "
              f"{r['delta_auroc']:+5.2f} | {r['correct_miou']:5.2f}% | "
              f"{r['correct_map']:5.2f}%")

    # Print LaTeX row
    print(f"\n{'='*70}")
    print("LaTeX row (paste into rebuttal Table 3):")
    print("─" * 70)
    auroc_vals = " & ".join(f"{r['correct_auroc']:.1f}" for r in results)
    delta_vals = " & ".join(f"{r['delta_auroc']:+.1f}" for r in results)
    print(f"\\textbf{{AUROC}} & {auroc_vals} \\\\")
    print(f"$\\Delta$ & {delta_vals} \\\\")
    print("─" * 70)

    # ── save ─────────────────────────────────────────────────────────
    output = {
        'model': model_label,
        'method': method_label,
        'baseline_auroc': baseline_auroc,
        'max_dict_cos_sim': args.max_dict_cos_sim,
        'sparse_threshold': args.sparse_threshold,
        'use_llm_dictionary': args.use_llm_dictionary,
        'use_gpt_dictionary': args.use_gpt_dictionary,
        'fix_dictionary': args.fix_dictionary,
        'n_images': args.limit,
        'results': results,
    }

    base, ext = os.path.splitext(args.output_json)
    suffix = f"_{method_label.lower()}_{model_label.lower()}"
    out_path = f"{base}{suffix}{ext}"

    with open(out_path, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
