from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

model_name_or_path = "./checkpoints/llava-v1.5-13b-pretrain"
    
class LlavaFGloraConfig(LlamaConfig):
    model_type = "llava_llama"
    def __init__(self, by_pass_hidden_size = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.by_pass_hidden_size = by_pass_hidden_size

    
config1 = LlavaFGloraConfig(by_pass_hidden_size = 1024, hidden_size = 1024)
config2 = LlavaFGloraConfig()

print(LlavaFGloraConfig().model_type)  # 输出: "llava_llama"
print(config1.model_type, config1.by_pass_hidden_size, config1.hidden_size)  # 输出: "llava_llama"