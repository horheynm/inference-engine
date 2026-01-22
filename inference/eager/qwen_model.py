from pathlib import Path
from transformers import AutoConfig
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open


model = "Qwen/Qwen3-0.6B"
config = AutoConfig.from_pretrained(model)

files = list_repo_files(model)
safetensors_files = [f for f in files if f.endswith(".safetensors")]

safetensor_paths = [
    hf_hub_download(repo_id=model, filename=f)  
    for f in safetensors_files
]

for file in safetensor_paths:
    with safe_open(file, "pt") as f: 
        for layer_weight in f.keys():
            breakpoint()
            print(f"layer: {layer_weight}\nweight: {f.get_tensor(layer_weight)}")
            # TODO: Replace the model arch weigths
