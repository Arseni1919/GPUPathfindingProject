import torch
import torch.nn as nn
from utils import plot_tensor, get_random_input, get_mask_from_map
import random
import time
import argparse
import os


class ClipWithFullGradients(torch.autograd.Function):
    """
    Forward: Clip values to [min_val, max_val]
    Backward: Pass gradients through UNCHANGED (preserves gradient magnitude)
    """
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        return torch.clamp(input, min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clip_preserve_grad(x, min_val=0.0, max_val=1.0):
    """Clips values but preserves full gradients"""
    return ClipWithFullGradients.apply(x, min_val, max_val)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='2D GPU-accelerated pathfinding')
    parser.add_argument('--map', type=str, default=None,
                       help='Name of map file (e.g., maze-32-32-2.map). If not provided, randomly selects from maps/2d/')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto)')
    return parser.parse_args()


class SimpleNN(nn.Module):
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
            if i % 500 == 0: 
                # plot_tensor(x, self.map_mask, self.start_xy, self.goal_xy, f"[{i}] Output after ReLU")
                print(f"Iteration {i}, max value: {x.max().item():.4f}")
            if x[0, 0, goal_y, goal_x] > 0:
                print(f"Goal reached at iteration {i}!")
                print("Saving expanded nodes visualization...")
                output_path = os.path.join('outputs', f'{self.map_name}_expanded_nodes.png')
                plot_tensor(x, self.map_mask.cpu(), self.start_xy, self.goal_xy,
                           f"{self.map_name} - Expanded Nodes (Iteration {i})", output_path)
                return x, True, saved_tensors
        return x, False, saved_tensors

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = random.randint(0, 10000)
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Using seed: {seed}")

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Select map file
    if args.map:
        map_path = os.path.join('maps', '2d', args.map)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")
        map_name = os.path.splitext(args.map)[0]  # Remove .map extension
    else:
        # Randomly select from available maps
        map_dir = os.path.join('maps', '2d')
        available_maps = [f for f in os.listdir(map_dir) if f.endswith('.map')]
        if not available_maps:
            raise FileNotFoundError("No .map files found in maps/2d/")
        selected_map = random.choice(available_maps)
        map_path = os.path.join(map_dir, selected_map)
        map_name = os.path.splitext(selected_map)[0]  # Remove .map extension
        print(f"Randomly selected map: {selected_map}")

    # Load map and initialize
    map_mask, height, width = get_mask_from_map(map_path)
    in_channels = 1
    out_channels = 1
    batch_size = 1

    # Move mask to device
    map_mask = map_mask.to(device)

    random_input, goal_tensor, start_xy, goal_xy = get_random_input(map_mask, batch_size, in_channels, height, width)
    print(f"Start coordinates: {start_xy}, Goal coordinates: {goal_xy}")

    random_input.requires_grad = True
    random_input = random_input.to(device)

    # Initialize model and move to device
    model = SimpleNN(in_channels, out_channels, map_mask, start_xy, goal_xy, map_name)
    model = model.to(device)

    # 4-connected kernel (up, down, left, right)
    conv1_kernel = torch.tensor([[[[ 0.,  1.,  0.],
                                   [ 1.,  0.,  1.],
                                   [ 0.,  1.,  0.]]]]).to(device)
    model.conv.weight.data = conv1_kernel
    model.conv.bias.data = torch.zeros(out_channels).to(device)

    # Forward pass
    print("\n=== Starting Forward Pass ===")
    start_time = time.time()
    output, goal_reached, saved_tensors = model(random_input)
    forward_time = time.time() - start_time
    print(f"Forward pass completed in {forward_time:.2f} seconds")

    if goal_reached:
        print("\n=== Starting Backward Pass ===")
        print("Computing gradients to trace optimal path...")
        start_backward = time.time()

        # Backward pass to trace path
        output[0, 0, goal_xy[0], goal_xy[1]].backward()
        all_activations = torch.cat(saved_tensors, dim=0)
        all_grads = torch.cat([t.grad for t in saved_tensors], dim=0)
        all_active_origins = all_activations * (all_grads > 0).float()
        final_tensor = all_active_origins.sum(dim=0)[0]
        backward_time = time.time() - start_backward
        print(f"Backward pass completed in {backward_time:.2f} seconds")

        # Save optimal path visualization
        print("Saving optimal path visualization...")
        output_path = os.path.join('outputs', f'{map_name}_optimal_paths.png')
        plot_tensor(final_tensor, map_mask.cpu(), start_xy, goal_xy,
                   f"{map_name} - Optimal Path", output_path)

        total_time = forward_time + backward_time
        print("\n=== SUCCESS ===")
        print(f"✓ Total execution time: {total_time:.2f} seconds (forward: {forward_time:.2f}s, backward: {backward_time:.2f}s)")
        print(f"✓ Two visualizations saved to outputs/:")
        print(f"  1. {map_name}_expanded_nodes.png - Activation state when goal was reached")
        print(f"  2. {map_name}_optimal_paths.png - Optimal path traced via gradient accumulation")
    else:
        print("\n=== FAILED ===")
        print("Goal was not reached during the forward pass.")