import asyncio
import collections
import io
import json
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import folder_paths
import nodes
import server
from server import BinaryEventTypes, PromptServer
import comfy.model_management
import execution


def test_send_image_async_offload():
    """Verify that send_image and send_image_with_metadata offload PIL compression

    via asyncio.to_thread, keeping event loop stalls minimal.
    """
    async def run_benchmark():
        srv = PromptServer.__new__(PromptServer)
        srv.sockets = {}
        srv.messages = asyncio.Queue()

        sent_messages = []
        async def mock_send_bytes(event, data, sid=None):
            sent_messages.append((event, data))
        srv.send_bytes = mock_send_bytes

        # Create a large test image (1024x1024) to induce measurable compression work
        img = Image.new("RGB", (1024, 1024), color=(128, 64, 32))

        # Background task that measures asyncio event loop responsiveness
        stalls = []
        stop_monitor = asyncio.Event()

        async def loop_monitor():
            while not stop_monitor.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(0.005)
                elapsed = time.perf_counter() - t0
                jitter = elapsed - 0.005
                stalls.append(jitter)

        monitor_task = asyncio.create_task(loop_monitor())

        # Send multiple preview images
        for _ in range(5):
            await srv.send_image(["PNG", img, None])
            await srv.send_image_with_metadata(["JPEG", img, None], {"step": 1, "prompt_id": "test_p"})

        stop_monitor.set()
        await monitor_task

        # Verify sent payloads
        assert len(sent_messages) == 10
        # Check PREVIEW_IMAGE (PNG has type_num 2)
        event, data = sent_messages[0]
        assert event == BinaryEventTypes.PREVIEW_IMAGE
        assert len(data) > 4
        # Check PREVIEW_IMAGE_WITH_METADATA
        event, data = sent_messages[1]
        assert event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA
        meta_len = int.from_bytes(data[:4], byteorder="big")
        meta_json = json.loads(data[4:4 + meta_len].decode("utf-8"))
        assert meta_json["step"] == 1
        assert meta_json["image_type"] == "image/jpeg"

        # Check maximum stall on the event loop during compression
        max_stall = max(stalls) if stalls else 0
        print(f"\nMax event loop stall during previews: {max_stall * 1000:.2f} ms")
        assert max_stall < 0.050, f"Event loop stalled excessively: {max_stall * 1000:.2f} ms"

    asyncio.run(run_benchmark())


def _find_route_handler(srv, path, method="GET"):
    for r in srv.routes:
        if r.path == path and r.method == method:
            return r.handler
    raise ValueError(f"Route {method} {path} not found on server")


def test_object_info_etag_and_caching():
    """Verify HTTP ETag generation, 304 Not Modified, and cache invalidation on the REAL /object_info route."""
    async def run_etag_test():
        loop = asyncio.get_running_loop()
        asset_mgr = MagicMock()
        asset_mgr.enabled = False
        srv = PromptServer(loop, asset_mgr)

        get_object_info = _find_route_handler(srv, "/object_info")

        # 1. Initial request without ETag -> returns 200 with new ETag and Cache-Control
        req = make_mocked_request("GET", "/object_info", app=srv.app)
        resp = await get_object_info(req)
        assert resp.status == 200
        assert "ETag" in resp.headers
        etag = resp.headers["ETag"]
        assert resp.headers["Cache-Control"] == "public, must-revalidate"
        data = json.loads(resp.text)
        assert isinstance(data, dict)
        assert len(data) > 0

        # Verify cached state on server
        assert srv._object_info_etag == etag
        assert srv._object_info_json == resp.text
        assert srv._cached_object_info == resp.text

        # 2. Subsequent request with matching If-None-Match -> returns 304 Not Modified
        req_etag = make_mocked_request("GET", "/object_info", headers={"If-None-Match": etag}, app=srv.app)
        resp_304 = await get_object_info(req_etag)
        assert resp_304.status == 304
        assert resp_304.headers["ETag"] == etag
        assert resp_304.body is None or len(resp_304.body) == 0

        # 3. Weak validators, whitespace, and wildcard matching
        for variant in [f"W/{etag}", f"W/ {etag}", f"w/{etag}", "*", f'"different", {etag}']:
            req_weak = make_mocked_request("GET", "/object_info", headers={"If-None-Match": variant}, app=srv.app)
            resp_weak = await get_object_info(req_weak)
            assert resp_weak.status == 304, f"Failed for If-None-Match variant: {variant}"

        # 4. Request with mismatched If-None-Match -> returns 200 with cached JSON
        req_mismatch = make_mocked_request("GET", "/object_info", headers={"If-None-Match": '"wrong_etag"'}, app=srv.app)
        resp_200 = await get_object_info(req_mismatch)
        assert resp_200.status == 200
        assert resp_200.text == srv._object_info_json

        # 5. Invalidation clears cache on server
        srv.invalidate_object_info_cache()
        assert srv._object_info_etag is None
        assert srv._object_info_json is None
        assert srv._cached_object_info is None

    asyncio.run(run_etag_test())


def test_dynamic_node_registration_cache_invalidation():
    """Verify that nodes.load_custom_node automatically invalidates PromptServer's object_info cache."""
    async def run_invalidation_test():
        loop = asyncio.get_running_loop()
        asset_mgr = MagicMock()
        asset_mgr.enabled = False
        srv = PromptServer(loop, asset_mgr)
        PromptServer.instance = srv

        try:
            srv._object_info_etag = 'test_etag'
            srv._object_info_json = '{"test": 1}'
            srv._cached_object_info = '{"test": 1}'

            # Simulate loading a custom node module
            mock_module = MagicMock()
            mock_module.NODE_CLASS_MAPPINGS = {"Phase4TestNode": MagicMock()}
            mock_module.NODE_DISPLAY_NAME_MAPPINGS = {}

            with patch("importlib.util.spec_from_file_location"), patch("importlib.util.module_from_spec", return_value=mock_module):
                with patch("os.path.isfile", return_value=True):
                    success = await nodes.load_custom_node("/mock/path/phase4_node.py")
                    assert success is True

            # Cache must now be invalidated
            assert srv._object_info_etag is None
            assert srv._object_info_json is None
            assert srv._cached_object_info is None
        finally:
            nodes.NODE_CLASS_MAPPINGS.pop("Phase4TestNode", None)
            PromptServer.instance = None

    asyncio.run(run_invalidation_test())


def test_transient_image_cache_and_view_route():
    """Verify transient image cache storage in SaveImage and RAM serving in the REAL /view route."""
    async def run_view_test():
        temp_dir = tempfile.mkdtemp()
        try:
            loop = asyncio.get_running_loop()
            asset_mgr = MagicMock()
            asset_mgr.enabled = False
            srv = PromptServer(loop, asset_mgr)
            PromptServer.instance = srv

            # Save an image with SaveImage
            test_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            with patch.object(folder_paths, "get_output_directory", return_value=temp_dir):
                with patch.object(folder_paths, "get_save_image_path", return_value=(temp_dir, "test_file", 1, "", "test_file")):
                    saver = nodes.SaveImage()
                    saver.output_dir = temp_dir
                    output = saver.save_images(test_tensor, filename_prefix="test_file")

            saved_item = output["ui"]["images"][0]
            filename = saved_item["filename"]
            expected_filepath = os.path.join(temp_dir, filename)

            # Check that it exists in the transient cache under normpath, abspath, and tuple key
            cached_norm = srv.get_transient_image(os.path.normpath(expected_filepath))
            assert cached_norm is not None
            image_bytes, content_type = cached_norm
            assert content_type == "image/png"
            assert len(image_bytes) > 0

            cached_abs = srv.get_transient_image(os.path.abspath(expected_filepath))
            assert cached_abs is not None

            cached_tuple = srv.get_transient_image((filename, "", "output"))
            assert cached_tuple is not None
            assert cached_tuple[0] == image_bytes

            # Delete the on-disk file to prove /view serves from RAM without reading disk
            if os.path.exists(expected_filepath):
                os.remove(expected_filepath)
            assert not os.path.exists(expected_filepath)

            view_handler = _find_route_handler(srv, "/view")

            with patch.object(folder_paths, "get_directory_by_type", return_value=temp_dir):
                req = make_mocked_request(
                    "GET",
                    f"/view?filename={filename}&subfolder=&type=output",
                    app=srv.app
                )
                resp = await view_handler(req)
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "image/png"
                assert resp.body == image_bytes
        finally:
            PromptServer.instance = None
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run_view_test())


def test_scoped_cancellation_manager():
    """Verify that cancelling prompt A does NOT interrupt or cancel prompt B."""
    cm = comfy.model_management.CancellationManager()

    # Register prompt A and B
    cm.register_prompt("prompt_A")
    cm.register_prompt("prompt_B")

    # Prompt A is running
    cm.set_current_prompt("prompt_A")
    assert not cm.is_cancelled()
    assert not cm.is_cancelled("prompt_A")
    assert not cm.is_cancelled("prompt_B")

    # Cancel prompt A specifically
    cm.cancel_prompt("prompt_A")
    assert cm.is_cancelled("prompt_A")
    assert cm.is_cancelled()  # Checks current prompt (prompt_A)
    assert not cm.is_cancelled("prompt_B")  # Prompt B is NOT cancelled!

    # Simulate model_management functions
    with patch.object(comfy.model_management, "cancellation_manager", cm):
        # When current prompt is A, processing_interrupted() is True
        assert comfy.model_management.processing_interrupted() is True
        assert comfy.model_management.processing_interrupted("prompt_A") is True
        assert comfy.model_management.processing_interrupted("prompt_B") is False

        with pytest.raises(comfy.model_management.InterruptProcessingException):
            comfy.model_management.throw_exception_if_processing_interrupted()

        # Prompt A unregisters upon completion
        cm.unregister_prompt("prompt_A")

        # Now Prompt B starts
        cm.set_current_prompt("prompt_B")
        assert not cm.is_cancelled()
        assert not cm.is_cancelled("prompt_B")
        assert comfy.model_management.processing_interrupted() is False
        assert comfy.model_management.processing_interrupted("prompt_B") is False

        # Should NOT raise an exception for prompt B
        comfy.model_management.throw_exception_if_processing_interrupted()

        # Prompt B completes successfully
        cm.unregister_prompt("prompt_B")
        assert cm.is_cancelled() is False

    # Test early cancellation before register_prompt
    cm_early = comfy.model_management.CancellationManager()
    cm_early.cancel_prompt("early_prompt")
    cm_early.register_prompt("early_prompt")
    # Early cancel flag must be preserved
    assert cm_early.is_cancelled("early_prompt") is True


def test_interrupt_endpoint_and_queue_volatile():
    """Verify PromptQueue interrupt_if_running uses CancellationManager and volatile queue."""
    mock_server = MagicMock()
    mock_server.queue_updated = MagicMock()
    pq = execution.PromptQueue(mock_server)

    item_a = (1, "prompt_A", {}, {}, [])
    item_b = (2, "prompt_B", {}, {}, [])

    with pq.mutex:
        pq.currently_running[1] = item_a

    # Interrupt running prompt A
    with patch.object(comfy.model_management.cancellation_manager, "cancel_prompt") as mock_cancel:
        res = pq.interrupt_if_running("prompt_A")
        assert res is True
        mock_cancel.assert_called_once_with("prompt_A")

    # Interrupt prompt B which is not running -> returns False
    with patch.object(comfy.model_management.cancellation_manager, "cancel_prompt") as mock_cancel:
        res = pq.interrupt_if_running("prompt_B")
        assert res is False
        mock_cancel.assert_not_called()

    # Volatile queue returns copy of current state without deepcopy
    running, queued = pq.get_current_queue_volatile()
    assert len(running) == 1
    assert running[0][1] == "prompt_A"

    # Queue delete cleans up token
    with pq.mutex:
        pq.queue.append(item_b)
    with patch.object(comfy.model_management.cancellation_manager, "unregister_prompt") as mock_unreg:
        pq.delete_queue_item(lambda x: x[1] == "prompt_B")
        mock_unreg.assert_called_once_with("prompt_B")
