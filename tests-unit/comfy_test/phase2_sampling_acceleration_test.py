import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from comfy.samplers import _calc_cond_batch, is_unit_multiplier, TeaCacheState
from comfy.model_patcher import ModelPatcher


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.device = torch.device("cpu")
        self.model_lowvram = False
        self.model_loaded_weight_memory = 64
        self.current_weight_patches_uuid = None
        self.current_patcher = None
        self._batch_schedule_cache = None

    def apply_model(self, x, t, **kwargs):
        return x

    def memory_required(self, input_shape, cond_shapes=None):
        return 100


class Phase2SamplingAccelerationTest(unittest.TestCase):
    def test_dynamic_batch_schedule_caching(self):
        # 1. Driver Query Stall Verification
        model = DummyModel()
        patcher = MagicMock()
        model.current_patcher = patcher

        mem_get_info_calls = 0
        def fake_get_free_memory(device):
            nonlocal mem_get_info_calls
            mem_get_info_calls += 1
            return 1000000

        patcher.get_free_memory.side_effect = fake_get_free_memory

        x_in = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor([1.0])

        cond_item = {
            'uuid': 'uuid_1',
            'model_conds': {},
        }
        conds = [[cond_item], [cond_item]]

        # Step 0: Should call get_free_memory and populate cache
        model_options_step0 = {
            'transformer_options': {'current_step': 0}
        }
        _calc_cond_batch(model, conds, x_in, timestep, model_options_step0)
        self.assertEqual(mem_get_info_calls, 1)
        self.assertIsNotNone(model._batch_schedule_cache)

        # Step 1: Same shape, step > 0 -> should reuse cache without querying driver
        model_options_step1 = {
            'transformer_options': {'current_step': 1}
        }
        _calc_cond_batch(model, conds, x_in, timestep, model_options_step1)
        self.assertEqual(mem_get_info_calls, 1, "Driver query stall: get_free_memory was called on step 1!")

        # Step 2: Same shape, step > 0 -> should reuse cache
        model_options_step2 = {
            'transformer_options': {'current_step': 2}
        }
        _calc_cond_batch(model, conds, x_in, timestep, model_options_step2)
        self.assertEqual(mem_get_info_calls, 1, "Driver query stall: get_free_memory was called on step 2!")

        # Step 0 again (new generation): Should invalidate and re-query
        _calc_cond_batch(model, conds, x_in, timestep, model_options_step0)
        self.assertEqual(mem_get_info_calls, 2)

        # Different shape / cache key: Should re-query
        x_in_different = torch.randn(4, 4, 8, 8)
        _calc_cond_batch(model, conds, x_in_different, timestep, model_options_step1)
        self.assertEqual(mem_get_info_calls, 3)

    def test_short_circuit_unweighted_cond_accumulation(self):
        # 2. Short-Circuit Unweighted Cond Accumulation
        model = DummyModel()
        patcher = MagicMock()
        patcher.get_free_memory.return_value = 1000000
        model.current_patcher = patcher

        x_in = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor([1.0])

        # Standard unweighted single conds
        cond_pos = {'uuid': 'uuid_p', 'model_conds': {}}
        cond_neg = {'uuid': 'uuid_n', 'model_conds': {}}
        conds = [[cond_pos], [cond_neg]]

        model_options = {'transformer_options': {'current_step': 1}}
        out = _calc_cond_batch(model, conds, x_in, timestep, model_options)
        self.assertEqual(len(out), 2)
        # For identity apply_model, out[0] and out[1] equal x_in
        self.assertTrue(torch.allclose(out[0], x_in))
        self.assertTrue(torch.allclose(out[1], x_in))

    def test_multiple_conds_weighted_accumulation(self):
        # When a cond list has 2 items for positive, it must average them
        model = DummyModel()
        patcher = MagicMock()
        patcher.get_free_memory.return_value = 1000000
        model.current_patcher = patcher

        x_in = torch.ones(1, 4, 8, 8)
        timestep = torch.tensor([1.0])

        cond_pos1 = {'uuid': 'uuid_p1', 'model_conds': {}}
        cond_pos2 = {'uuid': 'uuid_p2', 'model_conds': {}}
        cond_neg = {'uuid': 'uuid_n', 'model_conds': {}}
        conds = [[cond_pos1, cond_pos2], [cond_neg]]

        model_options = {'transformer_options': {'current_step': 1}}
        out = _calc_cond_batch(model, conds, x_in, timestep, model_options)
        self.assertEqual(len(out), 2)
        # out[0] is (x_in + x_in) / 2 = x_in
        self.assertTrue(torch.allclose(out[0], x_in))
        self.assertTrue(torch.allclose(out[1], x_in))

    def test_area_conditioning_accumulation(self):
        # When cond has an area mask, weights must be applied
        model = DummyModel()
        patcher = MagicMock()
        patcher.get_free_memory.return_value = 1000000
        model.current_patcher = patcher

        x_in = torch.ones(1, 4, 8, 8)
        timestep = torch.tensor([1.0])

        # Area is [height, width, y, x]
        cond_pos1 = {'uuid': 'uuid_p1', 'model_conds': {}}
        cond_pos2 = {'uuid': 'uuid_p2', 'model_conds': {}, 'area': (4, 4, 0, 0)}
        conds = [[cond_pos1, cond_pos2]]

        model_options = {'transformer_options': {'current_step': 1}}
        out = _calc_cond_batch(model, conds, x_in, timestep, model_options)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, x_in.shape)

    def test_unit_multiplier_check(self):
        self.assertTrue(is_unit_multiplier(1.0))
        self.assertTrue(is_unit_multiplier(1))
        self.assertFalse(is_unit_multiplier(0.9))
        self.assertTrue(is_unit_multiplier(torch.tensor(1.0)))
        self.assertFalse(is_unit_multiplier(torch.tensor(0.5)))
        self.assertFalse(is_unit_multiplier(torch.ones(1, 4, 8, 8)))

    def test_teacache_state(self):
        # 3. TeaCache State functionality & relative L1 calculation
        tc = TeaCacheState(threshold=0.3)
        v0 = torch.ones(1, 64)

        # First step: should always return False (initial baseline)
        skip0 = tc.should_skip_step(v0)
        self.assertFalse(skip0)
        self.assertEqual(tc.accumulated_rel_l1, 0.0)

        # Small modulation change (< 0.3): should skip
        v1 = v0 + 0.1  # rel diff is 0.1 / 1.0 = 0.1 < 0.3
        skip1 = tc.should_skip_step(v1)
        self.assertTrue(skip1)
        self.assertAlmostEqual(tc.accumulated_rel_l1, 0.1, places=3)

        # Another small change, accumulated exceeds threshold (0.1 + 0.25 = 0.35 >= 0.3)
        v2 = v1 + 0.15
        skip2 = tc.should_skip_step(v2)
        self.assertFalse(skip2)
        # Should reset accumulated rel l1 after triggering
        self.assertEqual(tc.accumulated_rel_l1, 0.0)

        # Reset
        tc.reset()
        self.assertIsNone(tc.previous_modulated_input)
        self.assertEqual(tc.accumulated_rel_l1, 0.0)

    def test_teacache_integration_in_model_patcher(self):
        model = DummyModel()
        mp = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        tc_state = mp.set_model_teacache(threshold=0.25)
        self.assertEqual(mp.model_options["teacache"].threshold, 0.25)
        self.assertEqual(mp.model_options["transformer_options"]["teacache"].threshold, 0.25)

        tc_state.accumulated_rel_l1 = 0.5
        mp.cleanup()
        self.assertEqual(tc_state.accumulated_rel_l1, 0.0)

    def test_inplace_lora_weight_deltas(self):
        # 4. In-Place LoRA Weight Delta Updates
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.linear.weight.data.fill_(1.0)
                self.device = torch.device("cpu")
                self.model_lowvram = False
                self.model_loaded_weight_memory = 16
                self.current_weight_patches_uuid = None

        net = TestNet()
        mp = ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))
        mp.load(torch.device("cpu"))

        # Verify residency
        self.assertTrue(mp.is_model_resident_on_device(torch.device("cpu")))

        # Add a weight patch (simulating LoRA)
        delta_weight = torch.full((4, 4), 0.5)
        mp.patches["linear.weight"] = [(1.0, (delta_weight,), 1.0, None, None)]
        mp.patches_uuid = "patch_1"

        # partially_load should update in-place without unpatch_model(offload_device)
        with patch.object(mp, 'unpatch_model', wraps=mp.unpatch_model) as mock_unpatch:
            mp.partially_load(torch.device("cpu"))
            # unpatch_model should NOT have been called with offload_device
            mock_unpatch.assert_not_called()

        # Weight should now be 1.0 + 0.5 = 1.5
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.5)))

        # Update LoRA strength / delta
        delta_weight2 = torch.full((4, 4), 0.2)
        mp.patches["linear.weight"] = [(1.0, (delta_weight2,), 1.0, None, None)]
        mp.patches_uuid = "patch_2"

        with patch.object(mp, 'unpatch_model', wraps=mp.unpatch_model) as mock_unpatch:
            mp.partially_load(torch.device("cpu"))
            mock_unpatch.assert_not_called()

        # Weight should now be 1.0 + 0.2 = 1.2
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.2)))

        # Remove patch
        mp.patches.clear()
        mp.patches_uuid = "clean"
        with patch.object(mp, 'unpatch_model', wraps=mp.unpatch_model) as mock_unpatch:
            mp.partially_load(torch.device("cpu"))
            mock_unpatch.assert_not_called()

        # Clean base weight 1.0 restored
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.0)))

    def test_multi_chunk_dynamic_batch_schedule_caching(self):
        # Verify that multi-chunk batches (where to_run is split across loop iterations)
        # cache all chunks in dict without overwriting, eliminating driver stalls on step > 0
        model = DummyModel()
        patcher = MagicMock()
        model.current_patcher = patcher

        calls = 0
        def fake_get_free_memory(dev):
            nonlocal calls
            calls += 1
            return 1000  # 1 item fits (600*1.5=900 < 1000), 2 items do not (1200*1.5=1800 > 1000)

        patcher.get_free_memory.side_effect = fake_get_free_memory
        model.memory_required = lambda input_shape, cond_shapes=None: input_shape[0] * 600

        x_in = torch.randn(1, 4, 8, 8)
        timestep = torch.tensor([1.0])
        cond1 = {"uuid": "u1", "model_conds": {}}
        cond2 = {"uuid": "u2", "model_conds": {}}
        conds = [[cond1], [cond2]]

        # Step 0: should make 2 calls (1 for each chunk)
        _calc_cond_batch(model, conds, x_in, timestep, {"transformer_options": {"current_step": 0}})
        self.assertEqual(calls, 2)

        # Step 1: should make 0 additional calls (both chunks hit dict cache)
        _calc_cond_batch(model, conds, x_in, timestep, {"transformer_options": {"current_step": 1}})
        self.assertEqual(calls, 2)

        # Step 2: should make 0 additional calls
        _calc_cond_batch(model, conds, x_in, timestep, {"transformer_options": {"current_step": 2}})
        self.assertEqual(calls, 2)

    def test_first_lora_on_resident_model(self):
        # Verify that applying the first LoRA to an already resident model updates in-place
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.linear.weight.data.fill_(1.0)
                self.device = torch.device("cpu")
                self.model_lowvram = False
                self.model_loaded_weight_memory = 16
                self.current_weight_patches_uuid = None

        net = TestNet()
        mp = ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))
        mp.load(torch.device("cpu"), full_load=True)
        # Model resident without prior patches applied
        mp.model.current_weight_patches_uuid = None

        delta_weight = torch.full((4, 4), 0.5)
        mp.patches["linear.weight"] = [(1.0, (delta_weight,), 1.0, None, None)]
        mp.patches_uuid = "patch_first_lora"

        mp.partially_load(torch.device("cpu"))
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.5)))

    def test_cuda_inplace_patch_preserves_device(self):
        # If CUDA is available, verify that in-place LoRA patching does not demote weights to CPU
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        cuda_dev = torch.device("cuda:0")
        cpu_dev = torch.device("cpu")

        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.linear.weight.data.fill_(1.0)
                self.device = torch.device("cpu")
                self.model_lowvram = False
                self.model_loaded_weight_memory = 16
                self.current_weight_patches_uuid = None

        net = TestNet()
        mp = ModelPatcher(net, cuda_dev, cpu_dev)
        delta_weight0 = torch.full((4, 4), 0.1)
        mp.patches["linear.weight"] = [(1.0, (delta_weight0,), 1.0, None, None)]
        mp.load(cuda_dev, full_load=True)
        self.assertEqual(net.linear.weight.device, cuda_dev)

        # Update patch via partially_load
        delta_weight1 = torch.full((4, 4), 0.5)
        mp.patches["linear.weight"] = [(1.0, (delta_weight1,), 1.0, None, None)]
        mp.patches_uuid = "patch_1"
        mp.partially_load(cuda_dev)

        self.assertEqual(net.linear.weight.device, cuda_dev)
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.5, device=cuda_dev)))

        # Remove patch via partially_load
        mp.patches.clear()
        mp.patches_uuid = "clean"
        mp.partially_load(cuda_dev)

        self.assertEqual(net.linear.weight.device, cuda_dev)
        self.assertTrue(torch.allclose(net.linear.weight, torch.full((4, 4), 1.0, device=cuda_dev)))

    def test_device_compatibility_residency_check(self):
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.device = torch.device("cpu")
                self.model_lowvram = False
                self.model_loaded_weight_memory = 16

        net = TestNet()
        mp = ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))
        mp.load(torch.device("cpu"), full_load=True)
        self.assertTrue(mp.is_model_resident_on_device(torch.device("cpu")))
        self.assertFalse(mp.is_model_resident_on_device(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")))


if __name__ == "__main__":
    unittest.main()
