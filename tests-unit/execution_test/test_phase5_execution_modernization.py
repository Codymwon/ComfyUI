import asyncio
import os
import sys
import tempfile
import threading
import time
import pytest
import torch

from comfy_execution.caching import BasicCache, ContentAddressedDiskCache
from comfy_execution.graph import (
    DynamicPrompt,
    ExecutionList,
    ResourceClass,
    classify_node_resource,
)
from execution import execute_ready_nodes
import comfy.storage


def test_ancestry_caching_benchmark():
    nodes_dict = {}
    for i in range(64):
        inputs = {}
        if i > 0:
            inputs["input1"] = [str(i - 1), 0]
        if i > 1:
            inputs["input2"] = [str(i - 2), 0]
        nodes_dict[str(i)] = {"class_type": "DummyNode", "inputs": inputs}

    dynprompt = DynamicPrompt(nodes_dict)
    cache = BasicCache()

    t0 = time.perf_counter()
    for node_id in dynprompt.nodes:
        ancestors, order_mapping = cache.get_ordered_ancestry(dynprompt, node_id)
        assert isinstance(ancestors, list)
        assert isinstance(order_mapping, dict)
    elapsed = (time.perf_counter() - t0) * 1000

    assert elapsed < 10.0, f"Ancestry traversal too slow: {elapsed:.2f} ms"

    # Second traversal should be pure O(1) cache hits
    t1 = time.perf_counter()
    for node_id in dynprompt.nodes:
        cache.get_ordered_ancestry(dynprompt, node_id)
    cached_elapsed = (time.perf_counter() - t1) * 1000
    assert cached_elapsed < 1.0, f"Memoized ancestry traversal too slow: {cached_elapsed:.2f} ms"


def test_ancestry_cache_invalidation_on_set_prompt():
    prompt = {"0": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}
    dynprompt = DynamicPrompt(prompt)
    cache = BasicCache()
    cache.get_ordered_ancestry(dynprompt, "0")
    assert "0" in cache._ancestry_memo

    class DummyIsChanged:
        async def get(self, nid): return "hash1"

    async def _run_set_prompt():
        await cache.set_prompt(dynprompt, ["0"], DummyIsChanged())
    asyncio.run(_run_set_prompt())

    assert cache._ancestry_memo == {}


def test_legacy_import_shim_compatibility():
    comfy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "comfy"))
    while comfy_dir in sys.path:
        sys.path.remove(comfy_dir)

    import nodes
    assert hasattr(nodes, "ComfyLegacyImportFinder")

    import utils
    import model_management
    import samplers
    import sd
    import diffusers_load

    assert utils.__name__ == "comfy.utils"
    assert model_management.__name__ == "comfy.model_management"
    assert samplers.__name__ == "comfy.samplers"
    assert sd.__name__ == "comfy.sd"
    assert diffusers_load.__name__ == "comfy.diffusers_load"

    assert hasattr(utils, "ProgressBar")
    assert hasattr(model_management, "get_torch_device")


def test_legacy_import_when_utils_preloaded():
    # Verify that even when utils package was imported beforehand,
    # installing shims correctly aliases comfy.utils and preserves root package submodules.
    import utils.extra_config
    import nodes
    nodes.install_legacy_import_shims()

    import utils
    assert utils.__name__ == "comfy.utils"
    assert hasattr(utils, "ProgressBar")
    assert hasattr(utils, "extra_config")


def test_cas_disk_cache_roundtrip():
    assert hasattr(comfy.storage, "ContentAddressedDiskCache")

    with tempfile.TemporaryDirectory() as tmpdir:
        cas = ContentAddressedDiskCache(tmpdir)
        key = cas.get_cache_key("CLIPTextEncode", {"text": "sunset over mountains"}, "weights_hash_abc")
        assert isinstance(key, str) and len(key) == 64

        tensors = {
            "embedding": torch.randn(1, 77, 768),
            "pooled": torch.randn(1, 768),
        }
        cas.store_intermediate(key, tensors)

        loaded = cas.load_intermediate(key)
        assert loaded is not None
        assert torch.allclose(tensors["embedding"], loaded["embedding"])
        assert torch.allclose(tensors["pooled"], loaded["pooled"])
        assert cas.load_intermediate("missing_key") is None


def test_cas_disk_cache_non_contiguous_tensors():
    with tempfile.TemporaryDirectory() as tmpdir:
        cas = ContentAddressedDiskCache(tmpdir)
        # Create a non-contiguous slice
        base = torch.randn(10, 10)
        non_contiguous_slice = base[::2, ::2]
        assert not non_contiguous_slice.is_contiguous()

        key = "non_contig_key"
        cas.store_intermediate(key, {"slice": non_contiguous_slice})

        loaded = cas.load_intermediate(key)
        assert loaded is not None
        assert torch.allclose(non_contiguous_slice, loaded["slice"])


def test_dual_queue_concurrency():
    assert classify_node_resource("CLIPTextEncode") == ResourceClass.CPU_IO_CONCURRENT
    assert classify_node_resource("LoadImage") == ResourceClass.CPU_IO_CONCURRENT
    assert classify_node_resource("KSampler") == ResourceClass.GPU_EXCLUSIVE
    assert classify_node_resource("VAEDecode") == ResourceClass.GPU_EXCLUSIVE
    assert classify_node_resource("") == ResourceClass.CPU_IO_CONCURRENT
    assert classify_node_resource(None) == ResourceClass.CPU_IO_CONCURRENT

    class MockPromptServer:
        def __init__(self):
            self.gpu_mutex = asyncio.Lock()

    async def _async_test():
        prompt = {
            "text_pos": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
            "text_neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg"}},
            "image": {"class_type": "LoadImage", "inputs": {"image": "img.png"}},
            "sampler": {"class_type": "KSampler", "inputs": {
                "positive": ["text_pos", 0],
                "negative": ["text_neg", 0],
                "image": ["image", 0],
            }},
        }
        dynprompt = DynamicPrompt(prompt)

        class DummyCache:
            def get_local(self, nid): return None
            def set_local(self, nid, val): pass

        execution_list = ExecutionList(dynprompt, DummyCache())
        for nid in prompt.keys():
            execution_list.add_node(nid)

        server = MockPromptServer()
        execution_log = []
        main_thread = threading.get_ident()

        def worker_fn(node_id):
            tid = threading.get_ident()
            t0 = time.perf_counter()
            execution_log.append((node_id, "start", t0, tid))
            time.sleep(0.04)
            t1 = time.perf_counter()
            execution_log.append((node_id, "end", t1, tid))
            return f"out_{node_id}"

        # Step 1: Ready nodes should be 3 CPU tasks
        ready = execution_list.get_ready_nodes()
        assert set(ready) == {"text_pos", "text_neg", "image"}

        results = await execute_ready_nodes(execution_list, server, execute_single_node=worker_fn)
        assert len(results) == 3

        starts = {e[0]: e[2] for e in execution_log if e[1] == "start"}
        ends = {e[0]: e[2] for e in execution_log if e[1] == "end"}
        threads = {e[0]: e[3] for e in execution_log if e[1] == "start"}

        for nid in ("text_pos", "text_neg", "image"):
            assert threads[nid] != main_thread, f"Node {nid} ran on main thread instead of worker thread"

        # Concurrency check (overlap)
        assert starts["text_pos"] < ends["text_neg"] and starts["text_neg"] < ends["text_pos"]

        for nid in ready:
            execution_list.pop_node(nid)

        # Step 2: Sampler becomes ready, executes exclusively under gpu_mutex
        ready2 = execution_list.get_ready_nodes()
        assert ready2 == ["sampler"]

        gpu_locked = False
        orig_acquire = server.gpu_mutex.acquire
        async def wrap_acquire():
            nonlocal gpu_locked
            gpu_locked = True
            return await orig_acquire()
        server.gpu_mutex.acquire = wrap_acquire

        results2 = await execute_ready_nodes(execution_list, server, execute_single_node=worker_fn)
        assert len(results2) == 1
        assert gpu_locked

    asyncio.run(_async_test())


def test_pipelined_cpu_gpu_overlap():
    # Test pipelined concurrency: ready nodes has BOTH a CPU task and a GPU task simultaneously
    class MockPromptServer:
        def __init__(self):
            self.gpu_mutex = asyncio.Lock()

    async def _async_test():
        prompt = {
            "cpu_node": {"class_type": "LoadImage", "inputs": {}},
            "gpu_node": {"class_type": "KSampler", "inputs": {}},
        }
        dynprompt = DynamicPrompt(prompt)

        class DummyCache:
            def get_local(self, nid): return None
            def set_local(self, nid, val): pass

        execution_list = ExecutionList(dynprompt, DummyCache())
        execution_list.add_node("cpu_node")
        execution_list.add_node("gpu_node")

        server = MockPromptServer()
        execution_log = []

        def worker_fn(node_id):
            tid = threading.get_ident()
            t0 = time.perf_counter()
            execution_log.append((node_id, "start", t0, tid))
            time.sleep(0.05)
            t1 = time.perf_counter()
            execution_log.append((node_id, "end", t1, tid))
            return f"out_{node_id}"

        results = await execute_ready_nodes(execution_list, server, execute_single_node=worker_fn)
        assert len(results) == 2

        starts = {e[0]: e[2] for e in execution_log if e[1] == "start"}
        ends = {e[0]: e[2] for e in execution_log if e[1] == "end"}

        # CPU and GPU tasks MUST overlap in time (pipelined)
        assert starts["gpu_node"] < ends["cpu_node"], "GPU task did not overlap with CPU task!"
        assert starts["cpu_node"] < ends["gpu_node"], "CPU task did not overlap with GPU task!"

    asyncio.run(_async_test())
