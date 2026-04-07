import torch
import torch.nn as nn
from utils3D import plot_voxel, get_random_input_3d, get_mask_from_3dmap, create_3d_kernel_18connected, visualize_start_goal_preview
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
    parser = argparse.ArgumentParser(description='3D GPU-accelerated pathfinding')
    parser.add_argument('--map', type=str, default=None,
                       help='Name of 3D map file (e.g., A1.3dmap). If not provided, randomly selects from maps/3d/')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Save every Nth iteration (default: 100, lower = more memory)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto)')
    return parser.parse_args()


class SimpleNN3D(nn.Module):
    def __init__(self, in_channels, out_channels, map_mask, start_xyz, goal_xyz, map_name, save_freq=100):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.map_mask = map_mask
        # Expand mask from (D, H, W) to (1, 1, D, H, W)
        self.map_mask_5d = map_mask.unsqueeze(0).unsqueeze(0)
        self.start_xyz = start_xyz
        self.goal_xyz = goal_xyz
        self.map_name = map_name
        self.save_freq = save_freq  # Save every Nth iteration to reduce memory

    def forward(self, x):
        saved_tensors = []
        goal_z, goal_y, goal_x = self.goal_xyz
        for i in range(int(1e6)):
            x = self.relu(self.conv(x) * self.map_mask_5d)
            # Clip values to prevent explosion while preserving full gradients
            x = clip_preserve_grad(x, min_val=0.0, max_val=1.0)

            # Only save periodically to reduce memory usage in 3D
            if i % self.save_freq == 0:
                x.retain_grad()
                saved_tensors.append(x)
            else:
                # Detach from computation graph to prevent memory buildup
                # This breaks the gradient chain but keeps the forward computation
                x = x.detach()
                x.requires_grad = True
            if i % 1 == 0:  # Change to 500 for less verbose output
                print(f"Iteration {i}, max value: {x.max().item():.4f}")
            if x[0, 0, goal_z, goal_y, goal_x] > 0:
                print(f"Goal reached at iteration {i}!")
                if i % self.save_freq != 0:
                    x.retain_grad()
                    saved_tensors.append(x)
                print("Saving expanded nodes visualization...")
                plot_voxel(x, self.map_mask.cpu(), self.start_xyz, self.goal_xyz,
                          f"{self.map_name}_expanded_nodes", output_dir='outputs')
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
        map_path = os.path.join('maps', '3d', args.map)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")
        map_name = os.path.splitext(args.map)[0]  # Remove .3dmap extension
    else:
        # Randomly select from available maps
        map_dir = os.path.join('maps', '3d')
        available_maps = [f for f in os.listdir(map_dir) if f.endswith('.3dmap')]
        if not available_maps:
            raise FileNotFoundError("No .3dmap files found in maps/3d/")
        selected_map = random.choice(available_maps)
        map_path = os.path.join(map_dir, selected_map)
        map_name = os.path.splitext(selected_map)[0]  # Remove .3dmap extension
        print(f"Randomly selected map: {selected_map}")

    # Load 3D map
    print(f"Loading 3D map: {map_path}")
    map_mask, depth, height, width = get_mask_from_3dmap(map_path)
    print(f"Map dimensions: {depth} x {height} x {width} (D x H x W)")
    print(f"Walkable voxels: {(map_mask == 1.0).sum().item()}")
    print(f"Occupied voxels: {(map_mask == 0.0).sum().item()}")

    in_channels = 1
    out_channels = 1
    batch_size = 1

    # Move mask to device
    map_mask = map_mask.to(device)

    # Generate random start and goal
    random_input, goal_tensor, start_xyz, goal_xyz = get_random_input_3d(
        map_mask, batch_size, in_channels, depth, height, width
    )
    print(f"Start coordinates (z,y,x): {start_xyz}")
    print(f"Goal coordinates (z,y,x): {goal_xyz}")

    # Visualize start and goal with obstacles
    preview_file = f'outputs/{map_name}_start_goal_preview.html'
    distance = visualize_start_goal_preview(map_mask, start_xyz, goal_xyz, output_file=preview_file)

    print("\n" + "=" * 70)
    print("WAITING FOR APPROVAL")
    print("=" * 70)
    print(f"Please open '{preview_file}' in your browser and verify:")
    print("  1. Start (green) and Goal (red) are visible")
    print("  2. Both are in walkable space (not inside obstacles)")
    print("  3. Distance seems reasonable")
    print("\nPress ENTER to continue with pathfinding, or Ctrl+C to abort...")
    input()
    print("✓ Approved! Starting pathfinding...\n")

    random_input.requires_grad = True
    random_input = random_input.to(device)

    # Initialize model and move to device
    # save_freq: save every Nth iteration to reduce memory (100 = ~1% memory usage)
    # Note: Due to detaching between saves, only recent checkpoints will have gradients
    # For full gradient path visualization, use smaller save_freq (e.g., 10) if you have enough memory
    SAVE_FREQ = args.save_freq
    model = SimpleNN3D(in_channels, out_channels, map_mask, start_xyz, goal_xyz, map_name, save_freq=SAVE_FREQ)
    model = model.to(device)
    print(f"Memory optimization: saving every {SAVE_FREQ}th iteration")

    # Set 18-connected kernel
    conv3d_kernel = create_3d_kernel_18connected().to(device)
    print(f"Kernel shape: {conv3d_kernel.shape}")
    print(f"Kernel connectivity: {conv3d_kernel.sum().item()} neighbors")
    model.conv.weight.data = conv3d_kernel
    model.conv.bias.data = torch.zeros(out_channels).to(device)

    # Forward pass
    print("\nStarting forward pass...")
    start_time = time.time()
    output, goal_reached, saved_tensors = model(random_input)
    forward_time = time.time() - start_time
    print(f"Forward pass completed in {forward_time:.2f} seconds")

    if goal_reached:
        print("\n=== Goal was reached during the forward pass ===")

        # Backward pass
        print("\n=== Starting Backward Pass ===")
        print("Computing gradients to trace optimal path...")
        start_backward = time.time()
        output[0, 0, goal_xyz[0], goal_xyz[1], goal_xyz[2]].backward()
        backward_time = time.time() - start_backward
        print(f"Backward pass completed in {backward_time:.2f} seconds")

        # Accumulate gradients (only from saved checkpoints that have gradients)
        print(f"\nAccumulating gradients from {len(saved_tensors)} saved checkpoints...")

        # Filter out tensors without gradients (due to detaching for memory)
        tensors_with_grads = [(t, t.grad) for t in saved_tensors if t.grad is not None]

        if len(tensors_with_grads) == 0:
            print("Warning: No gradients available (all checkpoints were detached)")
            print("Skipping gradient accumulation visualization")
            final_tensor = None
        else:
            print(f"Found {len(tensors_with_grads)} checkpoints with gradients (recent ones)")
            all_activations = torch.cat([t for t, _ in tensors_with_grads], dim=0)
            all_grads = torch.cat([g for _, g in tensors_with_grads], dim=0)
            all_active_origins = all_activations * (all_grads > 0).float()
            final_tensor = all_active_origins.sum(dim=0)[0]

        if final_tensor is not None:
            print(f"Final tensor shape: {final_tensor.shape}")
            print(f"Final tensor max: {final_tensor.max().item():.4f}")
            print(f"Final tensor nonzero count: {(final_tensor > 0).sum().item()}")

            # Visualize final accumulated gradients (plot_voxel handles device->CPU transfer)
            print("Saving optimal path visualization...")
            plot_voxel(final_tensor, map_mask.cpu(), start_xyz, goal_xyz,
                      f"{map_name}_optimal_paths", threshold=0.01, max_voxels=10000, output_dir='outputs')

        total_time = forward_time + backward_time
        print("\n=== SUCCESS ===")
        print(f"✓ Total execution time: {total_time:.2f} seconds (forward: {forward_time:.2f}s, backward: {backward_time:.2f}s)")
        print(f"✓ Two visualizations saved to outputs/:")
        print(f"  1. {map_name}_expanded_nodes.html - Activation state when goal was reached")
        print(f"  2. {map_name}_optimal_paths.html - Optimal path traced via gradient accumulation")
    else:
        print("\n=== FAILED ===")
        print("Goal was NOT reached during the forward pass.")

    print(f"\nRandom seed: {seed}")
