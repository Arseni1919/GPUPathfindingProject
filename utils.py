import torch
import matplotlib.pyplot as plt
import os


def plot_tensor(tensor, map_mask, start_xy, goal_xy, title, output_path=None):
    """
    Visualize 2D tensor with start/goal markers.

    Args:
        tensor: The tensor to visualize
        map_mask: The map mask
        start_xy: Start position (y, x)
        goal_xy: Goal position (y, x)
        title: Plot title
        output_path: If provided, saves to file instead of displaying
    """
    if tensor.dim() == 4:
        final_im = tensor[0, 0].detach().cpu().numpy().copy()
    else:
        final_im = tensor.detach().cpu().numpy().copy()
    print(f"[PLOT] max value: {final_im.max():.4f}, min value: {final_im.min():.4f}")
    # final_im[final_im > 0] = 1
    plt.imshow(final_im, cmap='gray')
    plt.scatter([start_xy[1]], [start_xy[0]], c='green', marker='o', label='Start')
    plt.scatter([goal_xy[1]], [goal_xy[0]], c='red', marker='x', label='Goal')
    plt.title(title)
    plt.legend()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


def get_random_input(map_mask, batch_size, in_channels, height, width):
    start_tensor = torch.zeros(batch_size, in_channels, height, width)
    goal_tensor = torch.zeros(batch_size, in_channels, height, width)
    while True:
        start_xy = torch.randint(0, height, (1,)).item(), torch.randint(0, height, (1,)).item()
        goal_xy = torch.randint(0, height, (1,)).item(), torch.randint(0, height, (1,)).item()
        if map_mask[start_xy[0], start_xy[1]] == 1.0 and map_mask[goal_xy[0], goal_xy[1]] == 1.0:  # Ensure the random pixel is walkable
            break
    start_tensor[0, 0, start_xy[0], start_xy[1]] = 1.0
    goal_tensor[0, 0, goal_xy[0], goal_xy[1]] = 1.0
    return start_tensor, goal_tensor, start_xy, goal_xy


def get_mask_from_map(dir):
    """
    Reads a .map file and constructs a height by width tensor.

    Args:
        dir: Path to the .map file

    Returns:
        torch.Tensor: A tensor with 1 for dots (.) and 0 for other characters
    """
    with open(dir, 'r') as f:
        lines = f.readlines()

    # Parse the header to get height and width
    height = None
    width = None
    map_start_idx = None

    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith('height'):
            height = int(line.split()[1])
        elif line.startswith('width'):
            width = int(line.split()[1])
        elif line == 'map':
            map_start_idx = idx + 1
            break

    if height is None or width is None or map_start_idx is None:
        raise ValueError("Invalid .map file format")

    # Create the mask tensor
    mask = torch.zeros(height, width)

    # Parse the map data
    for i in range(height):
        if map_start_idx + i < len(lines):
            line = lines[map_start_idx + i].rstrip('\n')
            for j, char in enumerate(line):
                if j < width and char == '.':
                    mask[i, j] = 1.0

    return mask, height, width
