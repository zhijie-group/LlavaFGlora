import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):         # 返回一个函数
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))                    # .group(1)的作用是返回正则表达式匹配中第一个括号捕获组的内容，即提取的数字部分         
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        # for name, param in nn.Sequential(*modules).state_dict().items():
        #     print('***', name, param.size())
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


if __name__ == "__main__":
    class TestConfig:
        def __init__(self, mm_hidden_size, hidden_size, mm_projector_type):
            super().__init__()
            self.mm_hidden_size = mm_hidden_size
            self.hidden_size = hidden_size
            self.mm_projector_type = mm_projector_type
            
    config = TestConfig(mm_hidden_size = 1024, 
                        hidden_size = 5120, 
                        mm_projector_type = "mlp2x_gelu")
    
    mm_projector = build_vision_projector(config)
    
