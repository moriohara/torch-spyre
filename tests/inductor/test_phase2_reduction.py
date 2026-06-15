#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Phase 2: Reduction Operations output verification

This test validates reduction operation correctness by comparing Spyre outputs
against CPU outputs only for tensors whose layouts are comparable to CPU.
ElementArrangement is not asserted at the PyTorch level because it does not
propagate from torch-spyre back into PyTorch tensors, and tensors with
DL16_TO_FP32 layout are not directly compared against CPU tensors.
"""

import torch
import torch._dynamo as dynamo

# Reset dynamo
dynamo.reset()

device = "spyre:0"


def test_reduction_on_stick_normalizes_ea():
    """
    Test reduction on stick dimension by verifying output correctness.
    """
    @torch.compile
    def convert_and_reduce(x):
        x_fp32 = x.to(torch.float32)  # EA: STANDARD → DL16_TO_FP32
        x_mean = x_fp32.mean(-1, keepdim=True)  # EA: DL16_TO_FP32 → STANDARD
        return x_mean
    
    # Create FP16 tensor
    x_fp16 = torch.randn((4, 64), dtype=torch.float16, device=device)
    
    # Run compiled function on Spyre
    result_spyre = convert_and_reduce(x_fp16)
    
    # Compare with CPU for correctness
    x_fp16_cpu = x_fp16.cpu()
    result_cpu = convert_and_reduce(x_fp16_cpu)
    torch.testing.assert_close(result_spyre.cpu(), result_cpu, rtol=1e-3, atol=1e-3)
    
    print(f"✓ Reduction on stick dimension output matches CPU ✓")


def test_reduction_on_nonstick_preserves_ea():
    """
    Test reduction on non-stick dimension by verifying only the restored FP16
    output, since the intermediate FP32 tensor uses DL16_TO_FP32 layout.
    """
    @torch.compile
    def convert_reduce_and_restore(x):
        x_fp32 = x.to(torch.float32)  # EA: STANDARD → DL16_TO_FP32
        x_mean = x_fp32.mean(1, keepdim=True)  # EA: DL16_TO_FP32 → DL16_TO_FP32 (non-stick reduction)
        x_fp16 = x_mean.to(torch.float16)  # EA: DL16_TO_FP32 → STANDARD (reverse conversion)
        return x_mean, x_fp16
    
    # Create 3D FP16 tensor
    x_fp16 = torch.randn((4, 8, 64), dtype=torch.float16, device=device)
    
    # Run compiled function on Spyre
    x_mean_fp32_spyre, x_mean_fp16_spyre = convert_reduce_and_restore(x_fp16)
    
    # Compare FP16 output with CPU for correctness
    x_fp16_cpu = x_fp16.cpu()
    _, x_mean_fp16_cpu = convert_reduce_and_restore(x_fp16_cpu)
    torch.testing.assert_close(x_mean_fp16_spyre.cpu(), x_mean_fp16_cpu, rtol=1e-3, atol=1e-3)
    
    print(f"✓ Reduction on non-stick dimension FP16 output matches CPU ✓")


def test_reduction_sum_on_stick_normalizes_ea():
    """
    Test sum reduction on stick dimension by verifying output correctness.
    """
    @torch.compile
    def convert_and_sum(x):
        x_fp32 = x.to(torch.float32)
        x_sum = x_fp32.sum(-1, keepdim=True)
        return x_sum
    
    x_fp16 = torch.randn((4, 64), dtype=torch.float16, device=device)
    result_spyre = convert_and_sum(x_fp16)
    
    # Compare with CPU for correctness
    x_fp16_cpu = x_fp16.cpu()
    result_cpu = convert_and_sum(x_fp16_cpu)
    torch.testing.assert_close(result_spyre.cpu(), result_cpu, rtol=1e-2, atol=1e-2)
    
    print(f"✓ Sum reduction on stick dimension output matches CPU ✓")


def test_reduction_var_on_stick_normalizes_ea():
    """
    Test variance reduction on stick dimension by verifying output correctness.
    """
    @torch.compile
    def convert_and_var(x):
        x_fp32 = x.to(torch.float32)
        x_var = x_fp32.var(-1, keepdim=True)
        return x_var
    
    x_fp16 = torch.randn((4, 64), dtype=torch.float16, device=device)
    result_spyre = convert_and_var(x_fp16)
    
    # Compare with CPU for correctness
    x_fp16_cpu = x_fp16.cpu()
    result_cpu = convert_and_var(x_fp16_cpu)
    torch.testing.assert_close(result_spyre.cpu(), result_cpu, rtol=1e-2, atol=1e-2)
    
    print(f"✓ Variance reduction on stick dimension output matches CPU ✓")


def test_rmsnorm_variance_computation():
    """
    Test the variance computation part of RMSNorm by verifying only the final
    variance output.
    
    This is a key part of the RMSNorm use case from the implementation plan.
    Intermediate tensors use DL16_TO_FP32 layout and are not directly
    comparable with CPU tensors.
    """
    @torch.compile
    def rmsnorm_variance(x):
        x_fp32 = x.to(torch.float32)  # STANDARD → DL16_TO_FP32
        x_pow = x_fp32.pow(2)  # DL16_TO_FP32 → DL16_TO_FP32
        variance = x_pow.mean(-1, keepdim=True)  # DL16_TO_FP32 → STANDARD
        return x_fp32, x_pow, variance
    
    # Simulate RMSNorm variance computation
    hidden_states_fp16 = torch.randn((4, 64), dtype=torch.float16, device=device)
    x_fp32_spyre, x_pow_spyre, variance_spyre = rmsnorm_variance(hidden_states_fp16)
    
    # Compare only the final output with CPU for correctness.
    # Intermediate tensors use DL16_TO_FP32 layout and do not match CPU layout.
    hidden_states_fp16_cpu = hidden_states_fp16.cpu()
    _, _, variance_cpu = rmsnorm_variance(hidden_states_fp16_cpu)
    torch.testing.assert_close(variance_spyre.cpu(), variance_cpu, rtol=1e-3, atol=1e-3)
    
    print(f"✓ RMSNorm variance computation outputs match CPU ✓")


def test_multiple_reductions_preserve_ea():
    """
    Test multiple reductions by verifying only outputs with CPU-comparable
    layouts.
    """
    @torch.compile
    def multiple_reductions(x):
        x_fp32 = x.to(torch.float32)
        x_mean1 = x_fp32.mean(1, keepdim=True)  # Reduce dim 1 (not stick) - preserves DL16_TO_FP32
        x_mean2 = x_mean1.mean(0, keepdim=True)  # Reduce dim 0 (not stick) - preserves DL16_TO_FP32
        x_mean2_fp16 = x_mean2.to(torch.float16)  # Convert back - restores STANDARD
        x_mean3 = x_mean2.mean(-1, keepdim=True)  # Reduce stick dimension - normalizes to STANDARD
        return x_mean1, x_mean2, x_mean2_fp16, x_mean3
    
    # Create 4D tensor
    x_fp16 = torch.randn((2, 4, 8, 64), dtype=torch.float16, device=device)
    x_mean1_spyre, x_mean2_spyre, x_mean2_fp16_spyre, x_mean3_spyre = multiple_reductions(x_fp16)
    
    # Compare only outputs with CPU-comparable layouts for correctness.
    # Intermediate FP32 tensors use DL16_TO_FP32 layout and are not compared.
    x_fp16_cpu = x_fp16.cpu()
    _, _, x_mean2_fp16_cpu, x_mean3_cpu = multiple_reductions(x_fp16_cpu)
    torch.testing.assert_close(x_mean2_fp16_spyre.cpu(), x_mean2_fp16_cpu, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_mean3_spyre.cpu(), x_mean3_cpu, rtol=1e-3, atol=1e-3)
    
    print(f"✓ Multiple reductions outputs match CPU ✓")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2: Reduction Operations Output Verification Tests")
    print("=" * 70)
    
    try:
        test_reduction_on_stick_normalizes_ea()
        test_reduction_on_nonstick_preserves_ea()
        test_reduction_sum_on_stick_normalizes_ea()
        test_rmsnorm_variance_computation()
        test_multiple_reductions_preserve_ea()
        
        print("\n" + "=" * 70)
        print("✓ All Phase 2 tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise

# Made with Bob
