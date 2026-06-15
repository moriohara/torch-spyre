#!/usr/bin/env python3
"""Simple test to verify Phase 1 EA propagation fix."""

import torch
import torch._dynamo as dynamo
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout

# Reset dynamo
dynamo.reset()

device = "spyre:0"
shape = (4, 64)

print("="*80)
print("Phase 1 Simple Test: Single-arg pointwise EA preservation")
print("="*80)

@torch.compile
def convert_and_pow(x):
    x_fp32 = x.to(torch.float32)
    x_pow = x_fp32.pow(2)
    return x_pow

# Create FP16 tensor
x_fp16 = torch.randn(shape, dtype=torch.float16, device=device)
print(f"\nInput tensor: shape={x_fp16.shape}, dtype={x_fp16.dtype}")
input_layout = get_spyre_tensor_layout(x_fp16)
print(f"Input EA: {input_layout.element_arrangement}")

# Run compiled function
result = convert_and_pow(x_fp16)
print(f"\nOutput tensor: shape={result.shape}, dtype={result.dtype}")
output_layout = get_spyre_tensor_layout(result)
print(f"Output EA: {output_layout.element_arrangement}")
print(f"Output device_size: {list(output_layout.device_size)}")
print(f"Output stride_map: {list(output_layout.stride_map)}")

# Check if EA is DL16_TO_FP32
if output_layout.element_arrangement == ElementArrangement.DL16_TO_FP32:
    print("\n✓ SUCCESS: EA is DL16_TO_FP32 as expected!")
    exit(0)
else:
    print(f"\n✗ FAILURE: Expected DL16_TO_FP32, got {output_layout.element_arrangement}")
    print("\nDEBUG: Checking device_size to infer EA...")
    if list(output_layout.device_size)[0] == 2:
        print("  device_size[0] == 2, which suggests DL16_TO_FP32 structure")
        print("  The EA might not be properly set in the SpyreTensorLayout constructor")
    exit(1)

# Made with Bob
