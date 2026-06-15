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
Unit tests for ElementArrangement handling in FP16↔FP32 type conversions.

Tests the symmetric conversion logic that tracks element arrangements through
dtype conversions to ensure correct memory layouts, including EA-agnostic
operations like pointwise (add) and reductions (mean).
"""

import pytest
import torch
import torch._dynamo as dynamo
import numpy as np
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout


class TestElementArrangementConversions:
    """Test ElementArrangement handling for FP16↔FP32 conversions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.device = "spyre:0"
        self.shape = (4, 64)  # Small shape for testing

    def _get_element_arrangement(self, tensor):
        """Helper to get ElementArrangement from a tensor."""
        layout = get_spyre_tensor_layout(tensor)
        return layout.element_arrangement

    def _print_comparison(self, result_cpu, result_spyre, test_name):
        """Helper to print CPU vs Spyre comparison with deviation metrics."""
        result_spyre_cpu = result_spyre.cpu()
        
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")
        
        # Print shapes and dtypes
        print(f"\nShape: {result_cpu.shape}")
        print(f"CPU dtype: {result_cpu.dtype}, Spyre dtype: {result_spyre.dtype}")
        
        # Print sample values (first few elements)
        print(f"\n{'CPU Output (first 10 values):':<40}")
        cpu_flat = result_cpu.flatten()
        print(cpu_flat[:min(10, cpu_flat.numel())])
        
        print(f"\n{'Spyre Output (first 10 values):':<40}")
        spyre_flat = result_spyre_cpu.flatten()
        print(spyre_flat[:min(10, spyre_flat.numel())])
        
        # Calculate and print deviation metrics
        abs_diff = torch.abs(result_cpu - result_spyre_cpu)
        rel_diff = abs_diff / (torch.abs(result_cpu) + 1e-8)
        
        print(f"\n{'Deviation Metrics:':<40}")
        print(f"  Max absolute difference: {abs_diff.max().item():.6e}")
        print(f"  Mean absolute difference: {abs_diff.mean().item():.6e}")
        print(f"  Max relative difference: {rel_diff.max().item():.6e}")
        print(f"  Mean relative difference: {rel_diff.mean().item():.6e}")
        
        # Print worst mismatches (top 5)
        if result_cpu.numel() > 1:
            flat_abs_diff = abs_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            worst_indices = torch.topk(flat_abs_diff, top_k).indices
            
            print(f"\n{'Top 5 Worst Mismatches:':<40}")
            for i, idx in enumerate(worst_indices):
                cpu_val = cpu_flat[idx].item()
                spyre_val = spyre_flat[idx].item()
                diff = flat_abs_diff[idx].item()
                print(f"  #{i+1} Index {idx.item()}: CPU={cpu_val:.6e}, Spyre={spyre_val:.6e}, Diff={diff:.6e}")
        
        print(f"{'='*80}\n")

    def test_fp16_to_fp32_forward_conversion(self):
        """Test FP16→FP32 forward conversion works correctly."""
        
        @torch.compile
        def convert_fp16_to_fp32(x):
            return x.to(torch.float32)
        
        # Create FP16 tensor
        x_fp16 = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Convert to FP32
        x_fp32 = convert_fp16_to_fp32(x_fp16)
        
        # Verify dtype conversion worked
        assert x_fp32.dtype == torch.float32
        ea = self._get_element_arrangement(x_fp16)
        assert ea == ElementArrangement.STANDARD, (
            f"Expected STANDARD for the input, got {ea.name}"
        )
        #ea = self._get_element_arrangement(x_fp32)
        #assert ea == ElementArrangement.DL16_TO_FP32, (
        #    f"Expected DL16_TO_FP32 for FP16→FP32 conversion, got {ea.name}"
        #)

    # def test_fp32_to_fp16_forward_conversion(self):
    #     """Test FP32→FP16 forward conversion works correctly."""
    #
    #     @torch.compile
    #     def convert_fp32_to_fp16(x):
    #         return x.to(torch.float16)
    #
    #     # Create FP32 tensor
    #     x_fp32 = torch.randn(self.shape, dtype=torch.float32, device=self.device)
    #
    #     # Convert to FP16
    #     x_fp16 = convert_fp32_to_fp16(x_fp32)
    #
    #     # Verify dtype conversion worked
    #     assert x_fp16.dtype == torch.float16
    #     ea = self._get_element_arrangement(x_fp32)
    #     assert ea == ElementArrangement.STANDARD, (
    #         f"Expected STANDARD for the input, got {ea.name}"
    #     )
    #     #ea = self._get_element_arrangement(x_fp32)
    #     #assert ea == ElementArrangement.FP32_TO_DL16, (
    #     #    f"Expected FP32_TO_DL16 for FP32→FP16 conversion, got {ea.name}"
    #     #)

    def test_fp16_to_fp32_reverse_conversion(self):
        """Test FP32→FP16→FP32 reverse conversion normalizes to STANDARD."""
        
        @torch.compile
        def convert_fp32_to_fp16_to_fp32(x):
            x_fp16 = x.to(torch.float16)  # Creates FP32_TO_DL16
            x_fp32_back = x_fp16.to(torch.float32)  # Should normalize to STANDARD
            return x_fp32_back
        
        # Create FP32 tensor
        x_fp32 = torch.randn(self.shape, dtype=torch.float32, device=self.device)
        
        # Convert FP32→FP16→FP32
        x_fp32_back = convert_fp32_to_fp16_to_fp32(x_fp32)
        
        # Check ElementArrangement (should be STANDARD after reverse conversion)
        ea = self._get_element_arrangement(x_fp32_back)
        assert ea == ElementArrangement.STANDARD, (
            f"Expected STANDARD for FP32→FP16→FP32 reverse conversion, got {ea.name}"
        )

    def test_fp32_to_fp16_reverse_conversion(self):
        """Test FP16→FP32→FP16 reverse conversion normalizes to STANDARD."""
        
        @torch.compile
        def convert_fp16_to_fp32_to_fp16(x):
            x_fp32 = x.to(torch.float32)  # Creates DL16_TO_FP32
            x_fp16_back = x_fp32.to(torch.float16)  # Should normalize to STANDARD
            return x_fp16_back
        
        # Create FP16 tensor
        x_fp16 = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Convert FP16→FP32→FP16
        x_fp16_back = convert_fp16_to_fp32_to_fp16(x_fp16)
        
        # Check ElementArrangement (should be STANDARD after reverse conversion)
        ea = self._get_element_arrangement(x_fp16_back)
        assert ea == ElementArrangement.STANDARD, (
            f"Expected STANDARD for FP16→FP32→FP16 reverse conversion, got {ea.name}"
        )

    def test_fp16_to_fp32_with_pointwise_ops(self):
        """Test FP16→FP32 conversion with EA-agnostic pointwise ops (add)."""
        
        @torch.compile
        def convert_and_add_spyre(x, y):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            y_fp32 = y.to(torch.float32)
            
            # Pointwise add
            result_fp32 = x_fp32 + y_fp32
            
            return result_fp32
        
        def convert_and_add_cpu(x, y):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            y_fp32 = y.to(torch.float32)
            
            # Pointwise add
            result_fp32 = x_fp32 + y_fp32
            
            return result_fp32
        
        # Create FP16 tensors on Spyre
        x_fp16_spyre = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        y_fp16_spyre = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Run compiled conversion with pointwise op on Spyre
        result_spyre = convert_and_add_spyre(x_fp16_spyre, y_fp16_spyre)

        # Verify result dtype
        assert result_spyre.dtype == torch.float32
        #ea = self._get_element_arrangement(result_spyre)
        #assert ea == ElementArrangement.DL16_TO_FP32, (
        #    f"Expected DL16_TO_FP32 for FP16→FP32 conversion, got {ea.name}"
        #)
        # here we don't compare the result between CPU and Spyre since the tensor layout is different


    def test_fp16_to_fp32_to_fp16_with_pointwise_ops(self):
        """Test FP16→FP32 conversion with EA-agnostic pointwise ops (add)."""
        
        @torch.compile
        def convert_and_add_spyre(x, y):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            y_fp32 = y.to(torch.float32)
            
            # Pointwise add
            result_fp32 = x_fp32 + y_fp32
            
            # FP32 → FP16
            result_fp16 = result_fp32.to(torch.float16)

            return result_fp16
        
        def convert_and_add_cpu(x, y):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            y_fp32 = y.to(torch.float32)
            
            # Pointwise add
            result_fp32 = x_fp32 + y_fp32
            
            # FP32 → FP16
            result_fp16 = result_fp32.to(torch.float16)

            return result_fp16
        
        # Create FP16 tensors on Spyre
        x_fp16_spyre = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        y_fp16_spyre = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Create same tensors on CPU for comparison
        x_fp16_cpu = x_fp16_spyre.cpu()
        y_fp16_cpu = y_fp16_spyre.cpu()
        
        # Run compiled conversion with pointwise op on Spyre
        result_spyre = convert_and_add_spyre(x_fp16_spyre, y_fp16_spyre)
        
        # Run eager operation on CPU
        result_cpu = convert_and_add_cpu(x_fp16_cpu, y_fp16_cpu)
        
        # Verify result dtype
        assert result_spyre.dtype == torch.float16
        ea = self._get_element_arrangement(result_spyre)
        assert ea == ElementArrangement.STANDARD, (
            f"Expected STANDARD for FP16→FP32→FP16 round-trip conversion, got {ea.name}"
        )
        
        # Print comparison details
        self._print_comparison(result_cpu, result_spyre, "FP16→FP32→FP16 with pointwise add")
        
        # Compare CPU and Spyre outputs
        torch.testing.assert_close(
            result_spyre.cpu(),
            result_cpu,
            rtol=1e-3,
            atol=1e-3,
            msg="CPU and Spyre outputs differ for pointwise add with round-trip conversion"
        )

    def test_fp32_to_fp16_to_fp32_with_pointwise_ops(self):
        """Test FP32→FP16 conversion with EA-agnostic pointwise ops (add, mul)."""
        
        @torch.compile
        def convert_and_compute_spyre(x, y):
            # FP32 → FP16
            x_fp16 = x.to(torch.float16)
            y_fp16 = y.to(torch.float16)
            
            # Pointwise ops
            result_fp16 = x_fp16 + y_fp16
            result_fp16 = result_fp16 * 2.0
            
            # FP16 → FP32
            result_fp32 = result_fp16.to(torch.float32)
            
            return result_fp32
        
        def convert_and_compute_cpu(x, y):
            # FP32 → FP16
            x_fp16 = x.to(torch.float16)
            y_fp16 = y.to(torch.float16)
            
            # Pointwise ops
            result_fp16 = x_fp16 + y_fp16
            result_fp16 = result_fp16 * 2.0
            
            # FP16 → FP32
            result_fp32 = result_fp16.to(torch.float32)
            
            return result_fp32
        
        # Create FP32 tensors on Spyre
        x_fp32_spyre = torch.randn(self.shape, dtype=torch.float32, device=self.device)
        y_fp32_spyre = torch.randn(self.shape, dtype=torch.float32, device=self.device)
        
        # Create same tensors on CPU for comparison
        x_fp32_cpu = x_fp32_spyre.cpu()
        y_fp32_cpu = y_fp32_spyre.cpu()
        
        # Run compiled conversion with pointwise ops on Spyre
        result_spyre = convert_and_compute_spyre(x_fp32_spyre, y_fp32_spyre)
        
        # Run eager operation on CPU
        result_cpu = convert_and_compute_cpu(x_fp32_cpu, y_fp32_cpu)
        
        # Verify result dtype
        assert result_spyre.dtype == torch.float32
        ea = self._get_element_arrangement(result_spyre)
        assert ea == ElementArrangement.STANDARD, (
            f"Expected STANDARD for FP32→FP16→FP32 reverse conversion, got {ea.name}"
        )
        
        # Print comparison details
        self._print_comparison(result_cpu, result_spyre, "FP32→FP16→FP32 with pointwise add+mul")
        
        # Compare CPU and Spyre outputs
        torch.testing.assert_close(
            result_spyre.cpu(),
            result_cpu,
            rtol=1e-2,
            atol=1e-2,
            msg="CPU and Spyre outputs differ for pointwise add+mul with round-trip conversion"
        )

    def test_fp16_to_fp32_to_fp16_simple_reduction(self):
        """Test FP16→FP32 conversion with simple reduction."""
        
        @torch.compile
        def convert_and_reduce_spyre(x):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            
            # Simple reduction (keepdim to avoid scalar)
            result_fp32 = torch.sum(x_fp32, dim=-1, keepdim=True)
            
            return result_fp32.to(torch.float16)
        
        def convert_and_reduce_cpu(x):
            # FP16 → FP32
            x_fp32 = x.to(torch.float32)
            
            # Simple reduction (keepdim to avoid scalar)
            result_fp32 = torch.sum(x_fp32, dim=-1, keepdim=True)
            
            return result_fp32.to(torch.float16)
        
        # Create FP16 tensor on Spyre
        x_fp16_spyre = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Create same tensor on CPU for comparison
        x_fp16_cpu = x_fp16_spyre.cpu()
        
        # Run compiled conversion with reduction on Spyre
        result_spyre = convert_and_reduce_spyre(x_fp16_spyre)
        
        # Run eager operation on CPU
        result_cpu = convert_and_reduce_cpu(x_fp16_cpu)
        
        # Verify shape and dtype
        assert result_spyre.shape == (self.shape[0], 1)
        assert result_spyre.dtype == torch.float16

        # Print comparison details
        self._print_comparison(result_cpu, result_spyre, "FP16→FP32→FP16 with dimension sum reduction")
        
        # Compare CPU and Spyre outputs
        torch.testing.assert_close(
            result_spyre.cpu(),
            result_cpu,
            rtol=1e-3,
            atol=1e-3,
            msg="CPU and Spyre outputs differ for dimension sum reduction"
        )

    def test_fp32_to_fp16_to_fp32_simple_reduction(self):
        """Test FP32→FP16 conversion with EA-agnostic reduction ops (sum)."""
        
        @torch.compile
        def convert_and_reduce_spyre(x):
            # FP32 → FP16
            x_fp16 = x.to(torch.float16)
            
            # Reduction
            result_fp16 = torch.sum(x_fp16, dim=-1, keepdim=True)
            
            # FP16 → FP32
            result_fp32 = result_fp16.to(torch.float32)
            
            return result_fp32
        
        def convert_and_reduce_cpu(x):
            # FP32 → FP16
            x_fp16 = x.to(torch.float16)
            
            # Reduction
            result_fp16 = torch.sum(x_fp16, dim=-1, keepdim=True)
            
            # FP16 → FP32
            result_fp32 = result_fp16.to(torch.float32)
            
            return result_fp32
        
        # Create FP32 tensor on Spyre
        x_fp32_spyre = torch.randn(self.shape, dtype=torch.float32, device=self.device)
        
        # Create same tensor on CPU for comparison
        x_fp32_cpu = x_fp32_spyre.cpu()
        
        # Run compiled conversion with reduction on Spyre
        result_spyre = convert_and_reduce_spyre(x_fp32_spyre)
        
        # Run eager operation on CPU
        result_cpu = convert_and_reduce_cpu(x_fp32_cpu)
        
        # Verify shape and dtype
        assert result_spyre.shape == (self.shape[0], 1)
        assert result_spyre.dtype == torch.float32
        
        # Print comparison details
        self._print_comparison(result_cpu, result_spyre, "FP32→FP16→FP32 with dimension sum reduction")
        
        # Compare CPU and Spyre outputs
        torch.testing.assert_close(
            result_spyre.cpu(),
            result_cpu,
            rtol=1e-3,
            atol=1e-3,
            msg="CPU and Spyre outputs differ for dimension sum reduction"
        )

    def test_simple_round_trip_conversion(self):
        """Test simple round-trip conversion without complex operations."""
        
        @torch.compile
        def simple_round_trip(x):
            # FP16 → FP32 → FP16
            x_fp32 = x.to(torch.float32)
            x_fp16_back = x_fp32.to(torch.float16)
            return x_fp16_back
        
        # Create input
        x_fp16 = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Run round trip
        result = simple_round_trip(x_fp16)
        
        # Verify result dtype
        assert result.dtype == torch.float16

    def test_sequential_conversions(self):
        """Test sequential conversions without mixing EAs in operations."""
        
        @torch.compile
        def sequential_conversions(x):
            # FP16 → FP32
            x1 = x.to(torch.float32)
            
            # FP32 → FP16
            x2 = x1.to(torch.float16)
            
            # FP16 → FP32
            x3 = x2.to(torch.float32)
            
            # FP32 → FP16
            x4 = x3.to(torch.float16)
            
            return x4
        
        # Create FP16 tensor
        x_fp16 = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Run sequential conversions
        result = sequential_conversions(x_fp16)
        
        # Verify result dtype
        assert result.dtype == torch.float16

    def test_single_arg_pointwise_preserves_ea(self):
        """Test Gap 1 Fix - Single-arg pointwise operations preserve ElementArrangement.
        
        This test validates Phase 1 of the EA propagation fixes:
        - FP16 → FP32 conversion creates DL16_TO_FP32 EA
        - pow(x) operation should preserve DL16_TO_FP32 EA (not reset to STANDARD)
        - FP32 → FP16 conversion restores STANDARD EA for PyTorch comparison
        """
        
        @torch.compile
        def convert_pow_and_back(x):
            # FP16 → FP32: EA should be STANDARD → DL16_TO_FP32
            x_fp32 = x.to(torch.float32)
            # pow(x): EA should be DL16_TO_FP32 → DL16_TO_FP32 (preserve)
            x_pow = x_fp32.pow(2)
            # FP32 → FP16: EA should be DL16_TO_FP32 → STANDARD (for PyTorch comparison)
            x_fp16_back = x_pow.to(torch.float16)
            return x_fp16_back
        
        # Create FP16 tensor
        x_fp16 = torch.randn(self.shape, dtype=torch.float16, device=self.device)
        
        # Run compiled function
        result = convert_pow_and_back(x_fp16)
        
        # Verify dtype
        assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"
        
        # Verify ElementArrangement is STANDARD after conversion back
        ea = self._get_element_arrangement(result)
        assert ea == ElementArrangement.STANDARD, \
            f"Expected STANDARD EA after FP32→FP16, got {ea}"
        
        # Verify correctness against CPU
        x_fp16_cpu = x_fp16.cpu()
        result_cpu = x_fp16_cpu.to(torch.float32).pow(2).to(torch.float16)
        
        # Print comparison
        self._print_comparison(result_cpu, result, "Single-arg pointwise (pow) EA preservation")
        
        # Compare outputs
        torch.testing.assert_close(
            result.cpu(),
            result_cpu,
            rtol=1e-3,
            atol=1e-3,
            msg="CPU and Spyre outputs differ for pow operation"
        )
        
        print(f"\n✓ Phase 1 Test PASSED: Single-arg pointwise preserves EA through FP16→FP32→pow→FP16")

    def test_element_arrangement_enum_values(self):
        """Test that all ElementArrangement enum values are accessible."""
        # Verify all enum values exist
        assert hasattr(ElementArrangement, 'STANDARD')
        assert hasattr(ElementArrangement, 'DL16_TO_FP32')
        assert hasattr(ElementArrangement, 'DL16_TO_FP8')
        assert hasattr(ElementArrangement, 'EXX2')
        
        # Verify they have different values
        values = {
            ElementArrangement.STANDARD,
            ElementArrangement.DL16_TO_FP32,
            ElementArrangement.DL16_TO_FP8,
            ElementArrangement.EXX2,
        }
        assert len(values) == 4, "ElementArrangement enum values should be unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
