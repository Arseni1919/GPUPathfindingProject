import torch
import numpy as np
import plotly.graph_objects as go
import os


def create_3d_kernel_18connected():
    """
    Creates an 18-connected 3D kernel (3x3x3).

    18-connectivity includes:
    - 6 face neighbors (differ in 1 coordinate by ±1)
    - 12 edge neighbors (differ in 2 coordinates by ±1)
    - Excludes 8 corner neighbors (differ in all 3 coordinates by ±1)

    Returns:
        torch.Tensor: Shape (1, 1, 3, 3, 3) with 18 ones
    """
    kernel_3d = torch.tensor([[[
        # z = -1 layer (5 neighbors: 1 face + 4 edges)
        [[0., 1., 0.],
         [1., 1., 1.],
         [0., 1., 0.]],

        # z = 0 layer (8 neighbors: 4 faces + 4 edges)
        [[1., 1., 1.],
         [1., 0., 1.],
         [1., 1., 1.]],

        # z = 1 layer (5 neighbors: 1 face + 4 edges)
        [[0., 1., 0.],
         [1., 1., 1.],
         [0., 1., 0.]]
    ]]])

    return kernel_3d


def get_mask_from_3dmap(file_path):
    """
    Loads a .3dmap file and creates a 3D mask tensor.

    Format:
    - Line 1: "voxel width height depth"
    - Following lines: "x y z" coordinates of occupied voxels

    Args:
        file_path: Path to .3dmap file

    Returns:
        tuple: (mask, depth, height, width)
            - mask: torch.Tensor of shape (depth, height, width)
              where 1.0 = walkable, 0.0 = occupied
            - depth, height, width: int dimensions

    Note: Tensor indexing is [z, y, x] but .3dmap file uses (x, y, z) format
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse dimensions from first line
    first_line = lines[0].strip().split()
    if first_line[0] != 'voxel':
        raise ValueError("Invalid .3dmap format: first line should start with 'voxel'")

    width = int(first_line[1])
    height = int(first_line[2])
    depth = int(first_line[3])

    # Create dense mask (1 = walkable, 0 = occupied)
    mask = torch.ones(depth, height, width, dtype=torch.float32)

    # Parse occupied voxel coordinates
    for line in lines[1:]:
        coords = line.strip().split()
        if len(coords) == 3:
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
            # Bounds check
            if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                # Note: tensor indexing is [z, y, x]
                mask[z, y, x] = 0.0

    return mask, depth, height, width


def get_random_input_3d(map_mask, batch_size, in_channels, depth, height, width):
    """
    Generate random start and goal positions in 3D walkable space.

    Args:
        map_mask: torch.Tensor of shape (depth, height, width) with 1 = walkable
        batch_size: int
        in_channels: int
        depth, height, width: int dimensions

    Returns:
        tuple: (start_tensor, goal_tensor, start_xyz, goal_xyz)
            - start_tensor: shape (batch_size, in_channels, depth, height, width)
            - goal_tensor: shape (batch_size, in_channels, depth, height, width)
            - start_xyz: tuple (z, y, x) for tensor indexing
            - goal_xyz: tuple (z, y, x) for tensor indexing
    """
    start_tensor = torch.zeros(batch_size, in_channels, depth, height, width)
    goal_tensor = torch.zeros(batch_size, in_channels, depth, height, width)

    # Find random walkable positions
    max_attempts = 10000
    attempts = 0
    while attempts < max_attempts:
        # Generate random coordinates
        start_z = torch.randint(0, depth, (1,)).item()
        start_y = torch.randint(0, height, (1,)).item()
        start_x = torch.randint(0, width, (1,)).item()

        goal_z = torch.randint(0, depth, (1,)).item()
        goal_y = torch.randint(0, height, (1,)).item()
        goal_x = torch.randint(0, width, (1,)).item()

        # Check if both positions are walkable
        if (map_mask[start_z, start_y, start_x] == 1.0 and
            map_mask[goal_z, goal_y, goal_x] == 1.0):
            start_xyz = (start_z, start_y, start_x)
            goal_xyz = (goal_z, goal_y, goal_x)
            break

        attempts += 1

    if attempts >= max_attempts:
        raise ValueError("Could not find walkable start and goal positions after 10000 attempts")

    # Place activation at start and goal
    start_tensor[0, 0, start_z, start_y, start_x] = 1.0
    goal_tensor[0, 0, goal_z, goal_y, goal_x] = 1.0

    return start_tensor, goal_tensor, start_xyz, goal_xyz


def visualize_start_goal_preview(map_mask, start_xyz, goal_xyz, output_file='outputs/start_goal_preview.html'):
    """
    Create a 3D visualization showing start, goal, and sampled obstacles.

    Args:
        map_mask: torch.Tensor of shape (depth, height, width) on any device
        start_xyz: tuple (z, y, x) for start position
        goal_xyz: tuple (z, y, x) for goal position
        output_file: str, filename for HTML output

    Returns:
        float: Euclidean distance between start and goal
    """
    import math

    # Move to CPU for visualization
    map_mask_cpu = map_mask.cpu() if map_mask.is_cuda or map_mask.device.type == 'mps' else map_mask
    depth, height, width = map_mask_cpu.shape

    start_z, start_y, start_x = start_xyz
    goal_z, goal_y, goal_x = goal_xyz

    # Calculate distance
    distance = math.sqrt((goal_z - start_z)**2 + (goal_y - start_y)**2 + (goal_x - start_x)**2)

    print(f"Euclidean distance: {distance:.2f} voxels")
    print("Creating start/goal visualization...")

    # Sample obstacles
    occupied_indices = torch.nonzero(map_mask_cpu == 0.0)
    sample_rate = max(1, len(occupied_indices) // 5000)
    sampled_obstacles = occupied_indices[::sample_rate]
    print(f"Showing {len(sampled_obstacles):,} obstacle voxels (sampled from {len(occupied_indices):,})")

    # Create figure
    fig = go.Figure()

    # Add obstacle voxels
    if len(sampled_obstacles) > 0:
        obs_z = sampled_obstacles[:, 0].numpy()
        obs_y = sampled_obstacles[:, 1].numpy()
        obs_x = sampled_obstacles[:, 2].numpy()
        fig.add_trace(go.Scatter3d(
            x=obs_x, y=obs_y, z=obs_z,
            mode='markers',
            marker=dict(size=2, color='darkgray', opacity=0.6),
            name='Obstacles'
        ))

    # Add start (green)
    fig.add_trace(go.Scatter3d(
        x=[start_x], y=[start_y], z=[start_z],
        mode='markers',
        marker=dict(size=15, color='green', symbol='circle'),
        name='Start'
    ))

    # Add goal (red)
    fig.add_trace(go.Scatter3d(
        x=[goal_x], y=[goal_y], z=[goal_z],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name='Goal'
    ))

    # Add line connecting start to goal
    fig.add_trace(go.Scatter3d(
        x=[start_x, goal_x], y=[start_y, goal_y], z=[start_z, goal_z],
        mode='lines',
        line=dict(color='yellow', width=3, dash='dash'),
        name=f'Direct path ({distance:.1f} voxels)'
    ))

    # Update layout
    fig.update_layout(
        title=f'Start (green) and Goal (red) - Distance: {distance:.1f} voxels',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=[0, width]),
            yaxis=dict(range=[0, height]),
            zaxis=dict(range=[0, depth]),
            aspectmode='manual',
            aspectratio=dict(
                x=width/max(width, height, depth),
                y=height/max(width, height, depth),
                z=depth/max(width, height, depth)
            )
        ),
        width=1400,
        height=1000
    )

    # Save to HTML
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.write_html(output_file)
    print(f"Visualization saved to: {output_file}")

    return distance


def plot_voxel(tensor, map_mask, start_xyz, goal_xyz, title, threshold=0.1, max_voxels=5000, output_dir='outputs'):
    """
    Fast 3D activation visualization using Plotly.

    Args:
        tensor: torch.Tensor with activations (can be 5D or 3D)
        map_mask: torch.Tensor of shape (depth, height, width)
        start_xyz: tuple (z, y, x) for start position
        goal_xyz: tuple (z, y, x) for goal position
        title: str title for the plot
        threshold: float, only plot voxels with activation > threshold (default 0.1 for speed)
        max_voxels: int, maximum voxels to plot (auto-samples if more)
    """
    print(f"[PLOT] Generating visualization...")

    # Extract 3D tensor if 5D (batch, channels, d, h, w)
    if tensor.dim() == 5:
        activation = tensor[0, 0].detach().cpu().numpy()
    elif tensor.dim() == 3:
        activation = tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Expected 3D or 5D tensor, got {tensor.dim()}D")

    # Get activated voxels above threshold
    activated_indices = np.argwhere(activation > threshold)

    if len(activated_indices) == 0:
        print(f"[PLOT] Warning: No voxels above threshold {threshold}")
        threshold = 0.01
        activated_indices = np.argwhere(activation > threshold)
        if len(activated_indices) == 0:
            print(f"[PLOT] Still no voxels, skipping plot")
            return

    # Auto-sample if too many voxels
    if len(activated_indices) > max_voxels:
        sample_rate = len(activated_indices) // max_voxels
        activated_indices = activated_indices[::sample_rate]

    # Extract coordinates
    z_coords = activated_indices[:, 0]
    y_coords = activated_indices[:, 1]
    x_coords = activated_indices[:, 2]
    values = activation[z_coords, y_coords, x_coords]

    print(f"[PLOT] Plotting {len(activated_indices):,} voxels (threshold={threshold})")

    # Create figure
    fig = go.Figure()

    # Plot activated voxels (no hover text for speed)
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=2, color=values, colorscale='Viridis', opacity=0.8),
        name='Activations',
        hoverinfo='skip'  # Disable hover for speed
    ))

    # Add start and goal markers
    start_z, start_y, start_x = start_xyz
    goal_z, goal_y, goal_x = goal_xyz

    fig.add_trace(go.Scatter3d(
        x=[start_x], y=[start_y], z=[start_z],
        mode='markers',
        marker=dict(size=12, color='green'),
        name='Start'
    ))

    fig.add_trace(go.Scatter3d(
        x=[goal_x], y=[goal_y], z=[goal_z],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Goal'
    ))

    # Simple layout
    depth, height, width = map_mask.shape
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1200,
        height=900,
        showlegend=True
    )

    # Save to HTML
    os.makedirs(output_dir, exist_ok=True)
    clean_title = title.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    output_file = os.path.join(output_dir, f'{clean_title}.html')
    fig.write_html(output_file)
    print(f"[PLOT] Saved to: {output_file}")
