#!/usr/bin/env python3
"""
Benchmark script for 2D GPU pathfinding.

Runs each map 100 times and measures average runtime (forward + backward pass only).
Results are saved to benchmarks.txt for later reference.
"""

import torch
import torch.nn as nn
import time
import statistics
import random
import os

# Import from search2D_nn
from search2D_nn import ClipWithFullGradients, clip_preserve_grad
from utils import get_random_input, get_mask_from_map


class SimpleNNBenchmark(nn.Module):
    """
    Benchmark version of SimpleNN without visualization overhead.
    Same algorithm, but skips plot_tensor calls and minimizes printing.
    """
    def __init__(self, in_channels, out_channels, map_mask, start_xy, goal_xy, map_name):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.map_mask = map_mask
        self.map_mask_4d = map_mask.unsqueeze(0).unsqueeze(0)
        self.start_xy = start_xy
        self.goal_xy = goal_xy
        self.map_name = map_name

    def forward(self, x):
        saved_tensors = []
        goal_y, goal_x = self.goal_xy
        for i in range(int(1e6)):
            x = self.relu(self.conv(x) * self.map_mask_4d)
            # Clip values to prevent explosion while preserving full gradients
            x = clip_preserve_grad(x, min_val=0.0, max_val=1.0)
            x.retain_grad()
            saved_tensors.append(x)
            # No printing or visualization during benchmarking
            if x[0, 0, goal_y, goal_x] > 0:
                return x, True, saved_tensors
        return x, False, saved_tensors


def benchmark_map(map_name, num_runs=100, seed=42):
    """
    Benchmark a single map over multiple runs.

    Args:
        map_name: Name of the map (without .map extension)
        num_runs: Number of benchmark runs (default: 100)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (avg_time, std_time, all_times)
    """
    print(f"\nBenchmarking {map_name}...")
    print(f"  Loading map and initializing...")

    # Load map (once, not timed)
    map_path = os.path.join('maps', '2d', f'{map_name}.map')
    map_mask, height, width = get_mask_from_map(map_path)

    # Device selection (once)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"  Using device: {device}")
    map_mask = map_mask.to(device)

    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    times = []
    successful_runs = 0

    for run in range(num_runs):
        # Generate random start/goal (not timed, but consistent due to seed)
        random_input, goal_tensor, start_xy, goal_xy = get_random_input(
            map_mask, 1, 1, height, width
        )
        random_input.requires_grad = True
        random_input = random_input.to(device)

        # Initialize model
        model = SimpleNNBenchmark(1, 1, map_mask, start_xy, goal_xy, map_name)
        model = model.to(device)

        # 4-connected kernel (up, down, left, right)
        conv1_kernel = torch.tensor([[[[ 0.,  1.,  0.],
                                       [ 1.,  0.,  1.],
                                       [ 0.,  1.,  0.]]]]).to(device)
        model.conv.weight.data = conv1_kernel
        model.conv.bias.data = torch.zeros(1).to(device)

        # TIMED SECTION: Forward + Backward pass
        start_time = time.time()

        # Forward pass (no visualization)
        output, goal_reached, saved_tensors = model(random_input)

        if goal_reached:
            # Backward pass
            output[0, 0, goal_xy[0], goal_xy[1]].backward()
            all_activations = torch.cat(saved_tensors, dim=0)
            all_grads = torch.cat([t.grad for t in saved_tensors], dim=0)
            all_active_origins = all_activations * (all_grads > 0).float()
            final_tensor = all_active_origins.sum(dim=0)[0]

            elapsed = time.time() - start_time
            times.append(elapsed)
            successful_runs += 1
        else:
            # Goal not reached - skip this run
            elapsed = time.time() - start_time
            print(f"  Warning: Goal not reached in run {run+1} (skipping)")

        # Clean up to avoid memory accumulation
        del model, random_input, output, saved_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Progress indicator
        if (run + 1) % 10 == 0:
            print(f"  Progress: {run+1}/{num_runs} runs completed")

    if len(times) == 0:
        raise RuntimeError(f"No successful runs for {map_name}")

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"  Completed: {successful_runs}/{num_runs} successful runs")
    print(f"  Average: {avg_time:.4f}s, Std: {std_time:.4f}s")

    return avg_time, std_time, times


def main():
    """Main benchmarking function."""
    print("=" * 60)
    print("GPU Pathfinding Benchmark")
    print("=" * 60)

    # Maps to benchmark (must match the pics/ directory)
    maps = [
        'maze512-2-9',
        'maze512-8-2',
        'warehouse-20-40-10-2-2'
    ]

    # Detect device
    if torch.cuda.is_available():
        device_name = 'CUDA'
    elif torch.backends.mps.is_available():
        device_name = 'MPS'
    else:
        device_name = 'CPU'

    print(f"Device: {device_name}")
    print(f"Number of runs per map: 100")
    print(f"Seed: 42 (for reproducibility)")
    print("=" * 60)

    results = {}

    # Run benchmarks
    for map_name in maps:
        try:
            avg_time, std_time, all_times = benchmark_map(map_name, num_runs=100, seed=42)
            results[map_name] = {
                'avg': avg_time,
                'std': std_time,
                'times': all_times
            }
        except Exception as e:
            print(f"  Error benchmarking {map_name}: {e}")
            results[map_name] = None

    # Save results to benchmarks.txt
    print("\n" + "=" * 60)
    print("Saving results to benchmarks.txt...")

    with open('benchmarks.txt', 'w') as f:
        f.write("GPU Pathfinding Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Number of runs per map: 100\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"Seed: 42\n")
        f.write("=" * 60 + "\n\n")

        for map_name in maps:
            data = results.get(map_name)
            if data is None:
                f.write(f"Map: {map_name}\n")
                f.write(f"  Error: Benchmarking failed\n\n")
                continue

            f.write(f"Map: {map_name}\n")
            f.write(f"  Average time: {data['avg']:.4f}s\n")
            f.write(f"  Std deviation: {data['std']:.4f}s\n")
            f.write(f"  Min time: {min(data['times']):.4f}s\n")
            f.write(f"  Max time: {max(data['times']):.4f}s\n")
            f.write(f"  All times: {data['times']}\n")
            f.write("\n")

    print("Results saved to benchmarks.txt")
    print("\n" + "=" * 60)
    print("Benchmark Summary:")
    print("=" * 60)

    for map_name in maps:
        data = results.get(map_name)
        if data is None:
            print(f"{map_name:30s} - FAILED")
        else:
            print(f"{map_name:30s} - {data['avg']:.4f}s ± {data['std']:.4f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()
