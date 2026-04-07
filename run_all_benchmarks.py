#!/usr/bin/env python3
"""Run benchmarks for all maps and save results."""

import subprocess
import sys

maps = [
    'maze512-2-9',
    'maze512-8-2',
    'warehouse-20-40-10-2-2'
]

results = {}

print("Running benchmarks for all maps (100 runs each)...")
print("This may take several minutes...")
print("="*60)

for map_name in maps:
    print(f"\nBenchmarking {map_name}...")
    try:
        result = subprocess.run(
            ['uv', 'run', 'benchmark_single.py', map_name, '100'],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes max
        )

        # Parse output for final results
        output = result.stdout
        if "Final:" in output:
            final_line = [l for l in output.split('\n') if 'Final:' in l][0]
            # Extract time from "Final: X.XXXXs ± X.XXXXs"
            time_part = final_line.split('Final:')[1].strip()
            results[map_name] = time_part
            print(f"  Result: {time_part}")
        else:
            print(f"  ERROR: Could not parse results")
            results[map_name] = "ERROR"

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Benchmark took too long")
        results[map_name] = "TIMEOUT"
    except Exception as e:
        print(f"  ERROR: {e}")
        results[map_name] = f"ERROR: {e}"

print("\n" + "="*60)
print("All benchmarks completed!")
print("="*60)

# Save to benchmarks.txt
with open('benchmarks.txt', 'w') as f:
    f.write("GPU Pathfinding Benchmark Results\n")
    f.write("="*60 + "\n")
    f.write("Number of runs per map: 100\n")
    f.write("Device: MPS (Apple Silicon)\n")
    f.write("Seed: 42\n")
    f.write("="*60 + "\n\n")

    for map_name, result in results.items():
        f.write(f"Map: {map_name}\n")
        f.write(f"  Result: {result}\n")
        f.write("\n")

print("\nResults saved to benchmarks.txt")
print("\nSummary:")
for map_name, result in results.items():
    print(f"  {map_name}: {result}")
