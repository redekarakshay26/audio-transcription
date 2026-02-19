import torch

def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    idx   = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available"    : True,
        "name"         : props.name,
        "vram_total"   : f"{props.total_memory / 1024**3:.2f} GB",
        "cuda_version" : torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()),
        "device_index" : idx,
    }


print(get_gpu_info())