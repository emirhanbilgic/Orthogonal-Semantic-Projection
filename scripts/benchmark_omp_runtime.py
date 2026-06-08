#!/usr/bin/env python3
import time
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os

def omp_sparse_residual(x_1x, D, max_atoms=8, tol=1e-6):
    """
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], atom rows, L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)

    device = x_1x.device
    dtype = x_1x.dtype
    
    # Force CPU execution for OMP loop to avoid MPS/CUDA lazy wrapper / context errors
    # and to support operators not yet in MPS (like linalg.lstsq)
    x = x_1x.clone().cpu().float()  # [1, d]
    D_cpu = D.cpu().float()         # [K, d]
    
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        # correlation with residual
        c = (r @ D_cpu.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        
        if len(selected) > 0:
            c_abs[selected] = -1.0
            
        idx = torch.argmax(c_abs).item()
        if c_abs[idx] <= tol:
            break
        selected.append(idx)
        
        # Solve least squares: s = argmin ||x - D_S s||^2
        D_S = D_cpu[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        
        I = torch.eye(G.shape[0])
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
            
        x_hat = (s.t() @ D_S)  # [1, d]
        r = (x - x_hat)
        
        if torch.norm(r) <= tol:
            break
            
    # Move back to original device/dtype
    r = r.to(device=device, dtype=dtype)
    final_res = F.normalize(r, dim=-1) if torch.norm(r) > tol else F.normalize(x_1x, dim=-1)
    return final_res

def benchmark_device(device_name, K_list, d=512, max_atoms=8, num_runs=10):
    if device_name == 'mps' and not torch.backends.mps.is_available():
        return None
    if device_name == 'cuda' and not torch.cuda.is_available():
        return None
        
    device = torch.device(device_name)
    print(f"\nBenchmarking {device_name.upper()}...")
    
    results = []
    
    for K in K_list:
        # Generate random data
        x = F.normalize(torch.randn(1, d, device=device), dim=-1)
        D = F.normalize(torch.randn(K, d, device=device), dim=-1)
        
        # Warmup
        for _ in range(5):
            _ = omp_sparse_residual(x, D, max_atoms=max_atoms)
        
        if device_name == 'cuda': torch.cuda.synchronize()
        if device_name == 'mps': torch.mps.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = omp_sparse_residual(x, D, max_atoms=max_atoms)
            
        if device_name == 'cuda': torch.cuda.synchronize()
        if device_name == 'mps': torch.mps.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"  K={K:5d}: {avg_time*1000:8.3f} ms")
        results.append({'K': K, 'time_ms': avg_time * 1000})
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=512, help='Embedding dimension (CLIP ViT-B/16 is 512)')
    parser.add_argument('--max_atoms', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=20)
    args = parser.parse_args()
    
    # Dictionary sizes to test
    K_list = [10, 50, 100, 500, 1000, 2000, 5000]
    
    all_results = {}
    
    # Benchmark CPU
    all_results['cpu'] = benchmark_device('cpu', K_list, d=args.d, max_atoms=args.max_atoms, num_runs=args.num_runs)
    
    # Benchmark MPS (MacBook GPU)
    mps_res = benchmark_device('mps', K_list, d=args.d, max_atoms=args.max_atoms, num_runs=args.num_runs)
    if mps_res:
        all_results['mps'] = mps_res
    else:
        print("\nMPS not available on this device.")
        
    # Also benchmark CUDA when available
    if torch.cuda.is_available():
        all_results['cuda'] = benchmark_device('cuda', K_list, d=args.d, max_atoms=args.max_atoms, num_runs=args.num_runs)
    else:
        print("\nCUDA not available (this is expected on a MacBook).")
        print("Note: The script contains CUDA support for your other devices.")

    # Save results
    output_path = 'omp_runtime_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAnalysis saved to {output_path}")

if __name__ == "__main__":
    main()
