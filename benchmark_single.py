#!/usr/bin/env python3
"""
Single map benchmark with timeout and progress tracking.
Usage: python benchmark_single.py <map_name> [num_runs]
Example: python benchmark_single.py maze512-2-9 10
"""

import torch
import torch.nn as nn
import time
import statistics
import random
import os
import sys
import signal

# Import from search2D_nn
from search2D_nn import ClipWithFullGradients, clip_preserve_grad
from utils import get_random_input, get_mask_from_map


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Run timeout exceeded")


class SimpleNNBenchmark(nn.Module):
    """Benchmark version without visualization."""
    def __init__(self, in_channels, out_channels, map_mask, start_xy, goal_xy, map_name):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.map_mask = map_mask
        self.map_mask_4d = map_mask.unsqueeze(0).unsqueeze(0)
        self.start_xy = start_xy
        self.goal_xy = goal_xy
        self.map_name = map_name

    def forward(self, x, max_iterations=100000):
        """Forward with iteration limit."""
        saved_tensors = []
        goal_y, goal_x = self.goal_xy
        for i in range(max_iterations):
            x = self.relu(self.conv(x) * self.map_mask_4d)
            x = clip_preserve_grad(x, min_val=0.0, max_val=1.0)
            x.retain_grad()
            saved_tensors.append(x)
            if x[0, 0, goal_y, goal_x] > 0:
                return x, True, saved_tensors, i
        return x, False, saved_tensors, max_iterations


def benchmark_single_run(model, random_input, goal_xy, timeout_seconds=30):
    """Run single benchmark with timeout."""
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        start_time = time.time()

        # Forward pass
        output, goal_reached, saved_tensors, iterations = model(random_input, max_iterations=100000)

        if goal_reached:
            # Backward pass
            output[0, 0, goal_xy[0], goal_xy[1]].backward()
            all_activations = torch.cat(saved_tensors, dim=0)
            all_grads = torch.cat([t.grad for t in saved_tensors], dim=0)
            all_active_origins = all_activations * (all_grads > 0).float()
            final_tensor = all_active_origins.sum(dim=0)[0]

            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel alarm
            return elapsed, True, iterations
        else:
            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel alarm
            return elapsed, False, iterations

    except TimeoutException:
        signal.alarm(0)  # Cancel alarm
        return None, False, -1
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        print(f"    Error: {e}")
        return None, False, -1


def benchmark_map(map_name, num_runs=10, seed=42, timeout_per_run=30):
    """Benchmark a single map."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {map_name}")
    print(f"{'='*60}")
    print(f"Runs: {num_runs}, Timeout per run: {timeout_per_run}s")

    # Load map
    map_path = os.path.join('maps', '2d', f'{map_name}.map')
    if not os.path.exists(map_path):
        print(f"ERROR: Map file not found: {map_path}")
        return None, None, []

    print(f"Loading map from {map_path}...")
    map_mask, height, width = get_mask_from_map(map_path)
    print(f"Map size: {height}x{width}")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")
    map_mask = map_mask.to(device)

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    times = []
    iterations_list = []

    print(f"\nRunning {num_runs} benchmarks...")
    print("-" * 60)

    for run in range(num_runs):
        # Generate random start/goal
        random_input, goal_tensor, start_xy, goal_xy = get_random_input(
            map_mask, 1, 1, height, width
        )
        random_input.requires_grad = True
        random_input = random_input.to(device)

        # Initialize model
        model = SimpleNNBenchmark(1, 1, map_mask, start_xy, goal_xy, map_name)
        model = model.to(device)

        conv1_kernel = torch.tensor([[[[ 0.,  1.,  0.],
                                       [ 1.,  0.,  1.],
                                       [ 0.,  1.,  0.]]]]).to(device)
        model.conv.weight.data = conv1_kernel
        model.conv.bias.data = torch.zeros(1).to(device)

        # Run benchmark
        print(f"Run {run+1}/{num_runs}...", end=" ", flush=True)
        elapsed, success, iters = benchmark_single_run(model, random_input, goal_xy, timeout_per_run)

        if elapsed is not None and success:
            times.append(elapsed)
            iterations_list.append(iters)
            print(f"✓ {elapsed:.3f}s ({iters} iterations)")
        elif elapsed is None:
            print(f"✗ TIMEOUT (>{timeout_per_run}s)")
        else:
            print(f"✗ FAILED (goal not reached in {iters} iterations)")

        # Cleanup
        del model, random_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("-" * 60)

    if len(times) == 0:
        print("ERROR: No successful runs!")
        return None, None, []

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    avg_iters = statistics.mean(iterations_list)

    print(f"\nResults:")
    print(f"  Successful: {len(times)}/{num_runs}")
    print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  Min time: {min(times):.4f}s")
    print(f"  Max time: {max(times):.4f}s")
    print(f"  Avg iterations: {avg_iters:.1f}")

    return avg_time, std_time, times


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_single.py <map_name> [num_runs]")
        print("Example: python benchmark_single.py maze512-2-9 10")
        sys.exit(1)

    map_name = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    avg, std, times = benchmark_map(map_name, num_runs=num_runs, timeout_per_run=30)

    if avg is not None:
        print(f"\n{'='*60}")
        print(f"Final: {avg:.4f}s ± {std:.4f}s")
        print(f"{'='*60}")
