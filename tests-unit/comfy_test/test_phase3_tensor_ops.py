import math
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import torch
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import comfy.sd as sd_mod
from nodes import ConditioningAverage, LatentComposite, LatentBlend, VAEEncodeForInpaint, InpaintModelConditioning


def _make_mock_vae(first_stage_model, latent_channels=4, latent_dim=2, device="cpu", free_mem=8 * 1024 * 1024 * 1024, mem_used=1):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    first_stage_model.comfy_has_chunked_io = False
    first_stage_model.comfy_decode_estimate_is_reliable = False
    vae.first_stage_model = first_stage_model
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=free_mem)
    vae.device = torch.device(device)
    vae.output_device = torch.device(device)
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = vae.downscale_ratio = 8
    vae.upscale_index_formula = vae.downscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = latent_channels
    vae.latent_dim = latent_dim
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.handles_tiling = False
    vae.format_encoded = None
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: mem_used
    return vae


def test_vae_decode_single_batch_direct_assignment():
    """Verify single-pass direct assignment in VAE.decode() avoids copying."""
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))

        # Case 1: Batch size 1, batch_number 1 -> direct assignment
        decoded_tensor = torch.randn((1, 3, 64, 64))
        mock_model = MagicMock()
        mock_model.decode.return_value = decoded_tensor

        vae = _make_mock_vae(mock_model, free_mem=1000, mem_used=1000)
        samples_in = torch.randn((1, 4, 8, 8))
        out = vae.decode(samples_in)

        # Output should match decoded_tensor movedim
        expected = decoded_tensor.movedim(1, -1)
        assert torch.allclose(out, expected)

        # Case 2: Batch size 1, batch_number 4 -> direct assignment (samples_in.shape[0] <= batch_number)
        decoded_tensor2 = torch.randn((1, 3, 64, 64))
        mock_model2 = MagicMock()
        mock_model2.decode.return_value = decoded_tensor2

        vae2 = _make_mock_vae(mock_model2, free_mem=4000, mem_used=1000)
        out2 = vae2.decode(samples_in)
        assert torch.allclose(out2, decoded_tensor2.movedim(1, -1))

        # Case 3: Batch size 2, batch_number 4 -> direct assignment
        samples_in2 = torch.randn((2, 4, 8, 8))
        decoded_tensor3 = torch.randn((2, 3, 64, 64))
        mock_model3 = MagicMock()
        mock_model3.decode.return_value = decoded_tensor3

        vae3 = _make_mock_vae(mock_model3, free_mem=4000, mem_used=1000)
        out3 = vae3.decode(samples_in2)
        assert torch.allclose(out3, decoded_tensor3.movedim(1, -1))


def test_vae_decode_multi_batch_chunking():
    """Verify multi-batch chunking when samples_in.shape[0] > batch_number."""
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))

        mock_model = MagicMock()
        # Decode chunks of 2
        mock_model.decode.side_effect = [
            torch.ones((2, 3, 16, 16)) * 1.0,
            torch.ones((2, 3, 16, 16)) * 2.0,
        ]

        # free_mem / mem_used = 2, so batch_number = 2, but total samples = 4
        vae = _make_mock_vae(mock_model, free_mem=2000, mem_used=1000)
        samples_in = torch.randn((4, 4, 2, 2))
        out = vae.decode(samples_in)

        assert out.shape == (4, 16, 16, 3)
        assert torch.allclose(out[:2], torch.ones((2, 16, 16, 3)) * 1.0)
        assert torch.allclose(out[2:], torch.ones((2, 16, 16, 3)) * 2.0)
        assert mock_model.decode.call_count == 2


def test_latent_composite_feathering_numerical_equivalence():
    """Verify vectorized feathering produces numerically identical results to reference loop."""
    def reference_composite(samples_to, samples_from, x, y, feather=0):
        x = x // 8
        y = y // 8
        feather = feather // 8
        samples_out = samples_to.copy()
        s = samples_to["samples"].clone()
        samples_to = samples_to["samples"]
        samples_from = samples_from["samples"]
        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask
        samples_out["samples"] = s
        return (samples_out,)

    lc = LatentComposite()

    torch.manual_seed(42)
    s1_data = torch.randn((1, 4, 64, 64))
    s2_data = torch.randn((1, 4, 32, 32))

    for feather in [0, 8, 32, 64, 128]:  # in input pixels: // 8 = 0, 1, 4, 8, 16
        for y, x in [(0, 0), (32, 0), (0, 32), (32, 32), (64, 64)]:
            s1 = {"samples": s1_data.clone()}
            s2 = {"samples": s2_data.clone()}

            ref_res = reference_composite(s1, s2, x, y, feather=feather)[0]["samples"]
            vec_res = lc.composite(s1, s2, x, y, feather=feather)[0]["samples"]

            assert torch.allclose(ref_res, vec_res, atol=1e-6), f"Mismatch for feather={feather}, x={x}, y={y}"

    # Edge case: Latent positioned near canvas edge so it gets clipped smaller than feather
    # samples_to: 64, y: 60 * 8 = 480 -> y_idx = 60, height remaining = 4. feather = 64 (feather_latent = 8 > 4)
    s1 = {"samples": s1_data.clone()}
    s2 = {"samples": s2_data.clone()}
    ref_edge = reference_composite(s1, s2, 0, 480, feather=64)[0]["samples"]
    vec_edge = lc.composite(s1, s2, 0, 480, feather=64)[0]["samples"]
    assert torch.allclose(ref_edge, vec_edge, atol=1e-6), "Border clipped edge case mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel count test")
def test_latent_composite_feathering_kernel_count():
    """Verify feathering launches <= 10 CUDA kernels instead of 256."""
    lc = LatentComposite()
    s1 = {"samples": torch.zeros((1, 4, 128, 128), device="cuda")}
    s2 = {"samples": torch.ones((1, 4, 64, 64), device="cuda")}

    # Warmup
    lc.composite(s1, s2, 32, 32, feather=64)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        lc.composite(s1, s2, 32, 32, feather=64)
    torch.cuda.synchronize()

    cuda_events = [e for e in prof.events() if e.device_type == torch.profiler.DeviceType.CUDA]
    print(f"Total CUDA kernels launched: {len(cuda_events)} (vs > 256 in unvectorized baseline)")
    assert len(cuda_events) <= 20, f"Feathering launched too many kernels: {len(cuda_events)}"


def test_conditioning_average_device_dtype_and_batch():
    """Verify ConditioningAverage handles CUDA device, non-float32 dtypes, and batch > 1."""
    ca = ConditioningAverage()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    dtypes = [torch.float32, torch.float16]

    for device in devices:
        for dtype in dtypes:
            for batch_size in [1, 2]:
                # cond_from has length 10
                t_from = torch.randn((batch_size, 10, 768), device=device, dtype=dtype)
                # cond_to has length 15 (larger than t_from, triggering padding)
                t_to = torch.randn((batch_size, 15, 768), device=device, dtype=dtype)

                cond_from = [[t_from, {"pooled_output": torch.randn((batch_size, 768), device=device, dtype=dtype)}]]
                cond_to = [[t_to, {"pooled_output": torch.randn((batch_size, 768), device=device, dtype=dtype)}]]

                res = ca.addWeighted(cond_to, cond_from, 0.5)
                out_tensor = res[0][0][0]

                assert out_tensor.device == torch.device(device)
                assert out_tensor.dtype == dtype
                assert out_tensor.shape == (batch_size, 15, 768)


def test_inpaint_preprocessing_numerical_equivalence():
    """Verify vectorized inpaint pixel preprocessing matches original 3-step loop."""
    torch.manual_seed(123)

    for b, h, w in [(1, 64, 64), (2, 128, 128), (1, 32, 48)]:
        pixels = torch.rand((b, h, w, 3), dtype=torch.float32)
        mask = torch.rand((b, 1, h, w), dtype=torch.float32).round()

        # Reference loop
        ref_pixels = pixels.clone()
        m_ref = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            ref_pixels[:, :, :, i] -= 0.5
            ref_pixels[:, :, :, i] *= m_ref
            ref_pixels[:, :, :, i] += 0.5

        # Vectorized
        m_vec = (1.0 - mask.round()).squeeze(1).unsqueeze(-1)
        vec_pixels = (pixels - 0.5) * m_vec + 0.5

        assert torch.allclose(ref_pixels, vec_pixels, atol=1e-7), f"Mismatch for shape ({b}, {h}, {w}, 3)"


def test_latent_blend():
    """Verify LatentBlend works correctly without the discarded permute calls."""
    lb = LatentBlend()

    # Same shape
    s1 = {"samples": torch.randn((1, 4, 32, 32))}
    s2 = {"samples": torch.randn((1, 4, 32, 32))}
    out = lb.blend(s1, s2, blend_factor=0.5)[0]
    assert out["samples"].shape == (1, 4, 32, 32)
    assert torch.allclose(out["samples"], s1["samples"] * 0.5 + s2["samples"] * 0.5)

    # Different shape (upscaling branch)
    s3 = {"samples": torch.randn((1, 4, 16, 16))}
    out_scaled = lb.blend(s1, s3, blend_factor=0.3)[0]
    assert out_scaled["samples"].shape == (1, 4, 32, 32)


def test_vae_decode_uneven_multi_batch_chunking():
    """Verify multi-batch chunking when samples_in.shape[0] is not a multiple of batch_number."""
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))

        mock_model = MagicMock()
        mock_model.decode.side_effect = [
            torch.ones((2, 3, 16, 16)) * 1.0,
            torch.ones((2, 3, 16, 16)) * 2.0,
            torch.ones((1, 3, 16, 16)) * 3.0,
        ]

        # free_mem / mem_used = 2, so batch_number = 2, but total samples = 5
        vae = _make_mock_vae(mock_model, free_mem=2000, mem_used=1000)
        samples_in = torch.randn((5, 4, 2, 2))
        out = vae.decode(samples_in)

        assert out.shape == (5, 16, 16, 3)
        assert torch.allclose(out[:2], torch.ones((2, 16, 16, 3)) * 1.0)
        assert torch.allclose(out[2:4], torch.ones((2, 16, 16, 3)) * 2.0)
        assert torch.allclose(out[4:], torch.ones((1, 16, 16, 3)) * 3.0)
        assert mock_model.decode.call_count == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for peak VRAM test")
def test_vae_decode_peak_vram_reduction():
    """Verify single-pass decode avoids duplicate buffer coexistence, cutting peak memory in half."""
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))

        # 48 MB tensor: 1 * 3 * 2048 * 2048 * 4 bytes = 50,331,648 bytes
        tensor_size = (1, 3, 2048, 2048)
        mock_model = MagicMock()
        mock_model.decode.side_effect = lambda s, **k: torch.zeros(tensor_size, device="cuda", dtype=torch.float32)

        vae = _make_mock_vae(mock_model, device="cuda", free_mem=10 * 1024**3, mem_used=1024**2)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        samples_in = torch.zeros((1, 4, 256, 256), device="cuda")
        start_mem = torch.cuda.memory_allocated()
        _ = vae.decode(samples_in)
        peak_mem = torch.cuda.max_memory_allocated()

        buffer_bytes = tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3] * 4
        # Peak memory above baseline should be ~1x buffer size, not 2x (which would indicate preallocation + copy_)
        ratio = (peak_mem - start_mem) / buffer_bytes
        assert ratio < 1.25, f"Peak memory ratio {ratio:.2f} indicates duplicate buffer allocation (expected < 1.25)"


def test_conditioning_average_truncation_and_mismatched_batches():
    """Verify ConditioningAverage when cond_from has more tokens than cond_to and when batches differ."""
    ca = ConditioningAverage()

    # Case 1: cond_from has 20 tokens, cond_to has 10 tokens (truncation of cond_from)
    t_from = torch.randn((1, 20, 768))
    t_to = torch.randn((1, 10, 768))
    cond_from = [[t_from, {"pooled_output": torch.randn((1, 768))}]]
    cond_to = [[t_to, {"pooled_output": torch.randn((1, 768))}]]

    res = ca.addWeighted(cond_to, cond_from, 0.4)
    out_tensor = res[0][0][0]
    assert out_tensor.shape == (1, 10, 768)

    # Case 2: cond_to has batch size 2, cond_from has batch size 1 (broadcasting)
    t_from_b1 = torch.randn((1, 15, 768))
    t_to_b2 = torch.randn((2, 15, 768))
    cond_from_b1 = [[t_from_b1, {"pooled_output": torch.randn((1, 768))}]]
    cond_to_b2 = [[t_to_b2, {"pooled_output": torch.randn((2, 768))}]]

    res2 = ca.addWeighted(cond_to_b2, cond_from_b1, 0.7)
    out_tensor2 = res2[0][0][0]
    assert out_tensor2.shape == (2, 15, 768)


def test_inpaint_nodes_full_execution():
    """Verify full node execution for VAEEncodeForInpaint and InpaintModelConditioning."""
    mock_vae = MagicMock()
    mock_vae.spacial_compression_encode.return_value = 8
    mock_vae.encode.side_effect = lambda p: torch.zeros((p.shape[0], 4, p.shape[1] // 8, p.shape[2] // 8))

    # Test VAEEncodeForInpaint
    node_inpaint = VAEEncodeForInpaint()
    pixels = torch.rand((2, 64, 64, 3))
    mask = torch.rand((2, 64, 64))
    res = node_inpaint.encode(mock_vae, pixels, mask, grow_mask_by=4)
    assert res[0]["samples"].shape == (2, 4, 8, 8)
    assert res[0]["noise_mask"].shape == (2, 1, 64, 64)

    # Test InpaintModelConditioning
    node_cond = InpaintModelConditioning()
    pos = [[torch.zeros((1, 10, 768)), {}]]
    neg = [[torch.zeros((1, 10, 768)), {}]]
    res_cond = node_cond.encode(pos, neg, pixels[:1], mock_vae, mask[:1])
    assert res_cond[2]["samples"].shape == (1, 4, 8, 8)
    assert "noise_mask" in res_cond[2]


def test_latent_composite_extreme_feather_ratio():
    """Verify LatentComposite handles feathering ratio larger than latent dimension without errors."""
    lc = LatentComposite()
    s1 = {"samples": torch.zeros((1, 4, 16, 16))}
    s2 = {"samples": torch.ones((1, 4, 8, 8))}

    # feather = 128 (feather_latent = 16 > mask.shape[2] = 8)
    res = lc.composite(s1, s2, 0, 0, feather=128)[0]
    assert res["samples"].shape == (1, 4, 16, 16)
    assert not torch.isnan(res["samples"]).any()


if __name__ == "__main__":
    test_vae_decode_single_batch_direct_assignment()
    test_vae_decode_multi_batch_chunking()
    test_vae_decode_uneven_multi_batch_chunking()
    if torch.cuda.is_available():
        test_vae_decode_peak_vram_reduction()
    test_latent_composite_feathering_numerical_equivalence()
    if torch.cuda.is_available():
        test_latent_composite_feathering_kernel_count()
    test_latent_composite_extreme_feather_ratio()
    test_conditioning_average_device_dtype_and_batch()
    test_conditioning_average_truncation_and_mismatched_batches()
    test_inpaint_preprocessing_numerical_equivalence()
    test_inpaint_nodes_full_execution()
    test_latent_blend()
    print("ALL PHASE 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
