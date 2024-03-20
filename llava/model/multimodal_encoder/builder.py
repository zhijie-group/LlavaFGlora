import os
from .clip_encoder import CLIPVisionTower
from .visual_tokenizer import VisualTokenizer
from ..language_model.llava_llama_FGlora import LlavaFGloraConfig
def build_vision_tower(vision_tower_cfg, **kwargs):
    # 通过 getattr() 函数获取 vision_tower_cfg 对象中的 mm_vision_tower 属性。如果该属性不存在，则获取 vision_tower 属性
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None)) 
    vision_tower = "/liymai24/sjtu/bokai/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"  
    is_absolute_path_exists = os.path.exists(vision_tower)      # 检查 vision_tower 变量对应的路径是否存在      # openai/clip-vit-large-patch14-336     is_absolute_path_exists = False
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
