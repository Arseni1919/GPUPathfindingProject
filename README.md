# GPU-Accelerated Neural Network Pathfinding 🚀

A novel approach to pathfinding using GPU-accelerated neural networks with convolutional activation propagation and gradient-based path tracing.

## Core Innovation 💡

This project implements pathfinding as a neural network forward-backward pass:

**1. Forward Pass (Activation Propagation):**
- Place activation at start position
- Iteratively apply convolution (spreads to neighbors) + ReLU
- Multiply by map mask (blocks obstacles)
- Custom gradient-preserving clipping prevents numerical explosion
- Continue until activation reaches goal

**2. Backward Pass (Path Extraction):**
- Compute gradients from goal back to start
- Gradient flow reveals the path taken by activation
- Accumulate positive gradients to extract final path

**3. GPU Acceleration:**
- All operations run on GPU (CUDA/MPS/CPU fallback)
- Handles large 3D maps efficiently (tested on 896×390×255 voxels)

## Features ✨

- 🗺️ 2D pathfinding on grid maps (.map format)
- 🧊 3D pathfinding on voxel maps (.3dmap format)
- 📊 Interactive 3D visualizations using Plotly
- ⚡ Automatic device selection (CUDA > MPS > CPU)
- 🎲 Random map selection for testing
- ⚙️ Configurable memory/quality tradeoffs

## Installation 📦

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with CUDA or Apple Silicon (MPS) for best performance

### Setup with uv (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd GPUPathfindingProject

# Install dependencies using uv
uv sync

# Activate the environment (optional, uv run handles this automatically)
source .venv/bin/activate
```

### Alternative: Setup with pip
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch >= 2.11.0` (with CUDA/MPS support)
- `matplotlib >= 3.10.8`
- `plotly >= 6.6.0`
- `psutil >= 7.2.2`

## Usage 🎮

### 2D Pathfinding

**Run on a specific map:**
```bash
uv run search2D_nn.py --map maze-128-128-1.map
```

**Random map selection:**
```bash
uv run search2D_nn.py
```

**With custom seed and device:**
```bash
uv run search2D_nn.py --map warehouse-20-40-10-2-2.map --seed 8535 --device cuda
```

### 3D Pathfinding

**Run on a specific map:**
```bash
uv run search3D_nn.py --map A1.3dmap
```

**With custom settings:**
```bash
uv run search3D_nn.py --map A1.3dmap --seed 8535 --save-freq 50 --device cuda
```

> **Note:** If you've activated the virtual environment with `source .venv/bin/activate`, you can use `python` instead of `uv run`.

**Common Parameters:**
- `--map`: Map file name (from `maps/2d/` or `maps/3d/`)
- `--seed`: Random seed for reproducibility
- `--device`: Force device (`auto`/`cuda`/`mps`/`cpu`) - available for both 2D and 3D

**3D-Specific Parameters:**
- `--save-freq`: Save every Nth iteration (default: 100, lower = more memory, better path visualization)

## Project Structure 📁

```
GPUPathfindingProject/
├── maps/
│   ├── 2d/          # 2D grid maps (.map format)
│   └── 3d/          # 3D voxel maps (.3dmap format)
├── outputs/         # Generated visualizations (HTML/PNG)
├── search2D_nn.py   # 2D pathfinding script
├── search3D_nn.py   # 3D pathfinding script
├── utils.py         # 2D utilities (map loading, plotting)
├── utils3D.py       # 3D utilities (map loading, 3D visualization)
├── pyproject.toml   # uv project configuration
├── uv.lock          # uv lock file
├── requirements.txt # Pip-compatible requirements
└── README.md        # This file
```

## Map Formats 📋

### 2D Maps (.map format)
```
type octile
height 32
width 32
map
@@@@@@@@...
...
```
- `.` = walkable
- `@`, `T`, etc. = obstacles

### 3D Maps (.3dmap format)
```
voxel <width> <height> <depth>
<x> <y> <z>
...
```
- Lists occupied (obstacle) voxels
- All other voxels are walkable

## How It Works 🔬

### Custom Gradient Clipping
Standard `torch.clamp()` kills gradients at boundaries. We use a custom autograd function that clips forward values while passing gradients unchanged:

```python
class ClipWithFullGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        return torch.clamp(input, min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # Full gradient passthrough
```

### Connectivity Kernels
- **2D:** 4-connected (up, down, left, right)
- **3D:** 18-connected (6 faces + 12 edges, excludes 8 corners)

### Memory Optimization (3D)
For large 3D maps, saving every iteration is memory-prohibitive. We use checkpoint-based gradient accumulation:
- Save every Nth iteration (configurable via `--save-freq`)
- Detach intermediate tensors to break gradient chain
- Trade path visualization quality for memory efficiency

## Example Outputs 🎨

- **2D:** PNG images with path overlay
- **3D:** Interactive HTML with Plotly (rotate, zoom, inspect)
- All outputs saved to `outputs/` directory

## Performance Tips 🏎️

1. **Use GPU:** CUDA or MPS dramatically speeds up computation
2. **Adjust save frequency:** Higher `--save-freq` = less memory (100-1000 for 3D)
3. **Map size:** Larger maps take longer but are fully supported
4. **Device selection:** Let `auto` choose, or force with `--device`

## Technical Details 🔧

### Algorithm Overview
1. Initialize activation tensor with 1.0 at start position
2. Apply Conv2d/Conv3d with neighbor connectivity kernel
3. Apply ReLU activation
4. Multiply by map mask to zero out obstacles
5. Apply custom gradient-preserving clip to [0, 1]
6. Check if goal position has positive activation
7. If goal reached: backward pass from goal
8. Accumulate positive gradients across saved checkpoints
9. Visualize accumulated gradients as the discovered path

### Why This Approach?
- **Differentiable:** The entire pathfinding process is differentiable
- **Parallel:** GPU acceleration for massive parallelism
- **Flexible:** Easy to modify connectivity patterns or cost functions
- **Visualizable:** Gradient flow shows exactly how activation propagated

## Contributing 🤝

Contributions welcome! Areas for improvement:
- More connectivity patterns (8-connected 2D, 26-connected 3D)
- Multi-agent pathfinding (MAPF)
- Optimal subpath guarantees
- Benchmarking against A*/Dijkstra

## License 📄

[Add license information]

## Citation 📚

If you use this code in research, please cite:
[Add citation information]

---

Built with PyTorch 🔥 | Visualized with Plotly 📊 | Accelerated by GPUs 🚀
