from transformers import AutoConfig
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

from inference.models import Qwen3ForCausalLM

model = "Qwen/Qwen3-0.6B"
config = AutoConfig.from_pretrained(model)

files = list_repo_files(model)
safetensors_files = [f for f in files if f.endswith(".safetensors")]

safetensor_paths = [
    hf_hub_download(repo_id=model, filename=f) for f in safetensors_files
]


model = Qwen3ForCausalLM(config)

for file in safetensor_paths:
    with safe_open(file, "pt") as f:
        for layer_name in f.keys():
            # print(f"layer: {layer_name}\nweight: {f.get_tensor(layer_name)}")
            print(f"layer: {layer_name}")

            if layer_name == "lm_head.weight":
                param = model.get_parameter(layer_name)
                param.data.copy_(f.get_tensor("lm_head.weight"))
