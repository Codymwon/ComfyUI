# ComfyUI root utils package.
# Delegates attribute lookups to comfy.utils for legacy custom node compatibility.

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    import importlib
    mod = importlib.import_module("comfy.utils")
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module 'utils' has no attribute {name!r}")

def __dir__():
    import importlib
    mod = importlib.import_module("comfy.utils")
    return sorted(set(globals().keys()) | set(dir(mod)))
