#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from .llava_llama_FGlora import LlavaFGloraConfig

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def expand_mask(mask: torch.Tensor, dtype: torch.dtype, num_queries: int = 64, value_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, prompt_len = mask.size()
    expanded_mask = mask[:, None, :, None].expand(bsz, 1, prompt_len, value_len).to(dtype)         # [bsz, 1, prompt_len, value_len]
    inverted_mask = 1.0 - expanded_mask
    queries_mask = torch.zeros(bsz, 1, num_queries, value_len)                                     # [bsz, 1, prompt_len + num_queries, value_len]
    inverted_mask = torch.cat((inverted_mask, queries_mask), dim = 2)

    # 使用masked_fill函数，将掩码中的1替换为指定的填充值，以确保在后续的计算中被忽略
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)       # [bsz, 1, prompt_len + num_queries, value_len]

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(sample, spatial_shapes=[(8, 8)]):
    bs, c, h, w = sample.shape
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=sample.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points([(h, w)], sample.device)

    return reference_points, spatial_shapes, level_start_index

class TextQformerCrossAttention(nn.Module):
    def __init__(self, config: LlavaFGloraConfig):
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.pretraining_tp = config.pretraining_tp
        
        self.num_queries = config.num_queries
        self.by_pass_hidden_size = config.by_pass_hidden_size
        self.by_pass_head_dim = self.by_pass_hidden_size // self.num_heads
        self.save_mem = config.save_mem
        
        if self.save_mem:
            self.q_former_weights = nn.Linear(self.hidden_size, self.num_queries * self.num_heads, bias = config.attention_bias)
            self.q_former_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.by_pass_head_dim, bias = config.attention_bias)
        else:
            self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, self.num_heads * self.head_dim))
            self.q_former_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias) 
            self.q_former_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
            self.q_former_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
            self.q_former_o_proj = nn.Linear(self.num_heads * self.head_dim, self.by_pass_hidden_size, bias = config.attention_bias)
        
        self._init_rope()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
                      
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            
        bsz, seq_length, _ = hidden_states.size()
        
        # q-former部分：
        if self.save_mem:  
            attn_weights = self.q_former_weights(hidden_states).view(bsz, seq_length, self.num_queries, self.num_heads).transpose(1, 3)                            # (bsz, self.num_heads, num_queries, seq_length)
            value_states = self.q_former_v_proj(hidden_states).view(bsz, seq_length, self.num_key_value_heads, self.by_pass_head_dim).transpose(1, 2)              # (bsz, self.num_key_value_heads, seq_length, self.by_pass_head_dim)
            
            
            if attn_weights.size() != (bsz, self.num_heads, self.num_queries, seq_length):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, self.num_queries, seq_length)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, self.num_queries, seq_length):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, self.num_queries, seq_length)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                
            attn_weights /= math.sqrt(self.by_pass_head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)                                                 
            # (bsz, self.num_heads, num_queries, seq_length) * (bsz, self.num_key_value_heads, seq_length, self.by_pass_head_dim) = (bsz, self.num_key_value_heads, num_queries, self.by_pass_head_dim)
            
            if attn_output.size() != (bsz, self.num_heads, self.num_queries, self.by_pass_head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, self.num_queries, self.by_pass_head_dim)}, but is"
                    f" {attn_output.size()}"
                )
                
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, seq_length, self.by_pass_hidden_size)

            attn_output = self.q_former_o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value
        else:
            self.query_tokens = self.query_tokens.repeat(bsz, 1, 1)
            hidden_states = torch.cat((hidden_states, self.query_tokens), dim = 1)
            seq_length += self.num_queries
            
            query_states = self.q_former_q_proj(hidden_states)
            key_states = self.q_former_k_proj(hidden_states)
            value_states = self.q_former_v_proj(hidden_states) 
            
            query_states = query_states.view(bsz, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                
            past_key_value = (key_states, value_states) if use_cache else None
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)         # (bsz, self.num_heads, seq_length, self.head_dim) * (bsz, self.num_heads, self.head_dim, seq_length) = (bsz, self.num_heads, seq_length, seq_length)
        
            if attn_weights.size() != (bsz, self.num_heads, seq_length, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, seq_length, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, seq_length, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, seq_length, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)          # (bsz, self.num_heads, seq_length, kv_seq_len) * (bsz, self.num_key_value_heads, seq_length, self.head_dim) = (bsz, self.num_key_value_heads, seq_length, self.head_dim)

            if attn_output.size() != (bsz, self.num_heads, seq_length, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, seq_length, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
                
                
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, seq_length, self.hidden_size)

            if self.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
                o_proj_slices = self.q_former_o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
            else:
                attn_output = self.q_former_o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
            
            return attn_output[-self.num_queries:], attn_weights, past_key_value
        
class QueryImageCrossAttention(nn.Module):
    def __init__(self, config: LlavaFGloraConfig):
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.num_queries = config.num_queries
        self.by_pass_hidden_size = config.by_pass_hidden_size
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        self.spatial_shapes = config.spatial_shapes                 # e.g. spatial_shapes=[(12, 12), (24, 24), (48, 48), (96, 96)]
        self.im2col_step = config.im2col_step 
        
        self.proj_to_prompt = nn.Linear(self.hidden_size, self.by_pass_hidden_size)
        self.sampling_offsets = nn.Linear(self.by_pass_hidden_size, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(self.by_pass_hidden_size, self.num_heads * self.num_levels * self.num_points)
        self.image_v_proj = nn.Linear(config.mm_hidden_size, self.by_pass_hidden_size)
        self.cross_attn_1_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)
        self._init_rope()
        
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def image_expand_mask(self, mask: torch.Tensor, dtype: torch.dtype):
        spatial_shapes = torch.as_tensor(self.spatial_shapes, dtype=torch.long)
        image_feature_length = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()
        expanded_mask = mask[:, :, None, None, None].repeat(1, 1, 1, 1, image_feature_length).to(dtype)         # [bsz, num_images, 1, 1, image_feature_length]
        inverted_mask = 1.0 - expanded_mask

        # 使用masked_fill函数，将掩码中的1替换为指定的填充值，以确保在后续的计算中被忽略
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)                  # [bsz, num_images, 1, 1, image_feature_length]
    
    
    def forward(
        self,                                           # n_images这里是每个batch里样本最大数量的图片数量, 有些sample是通过pad得到的
        hidden_states: torch.Tensor,                    # 包含了coarse_images的一些信息，用于作为prompt，让by_pass_hidden_states去提取相关的fine_grained的信息
        by_pass_hidden_states: torch.Tensor,
        multi_scale_images_features: torch.Tensor,      # torch.Size(bsz, n_images, \sum_{l=0}^{L-1} H_l \cdot W_l, mm)
        multi_scale_images_shape: torch.Tensor,         # torch.Size(n_images, n_levels)
        image_positions: torch.Tensor,                  # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
        image_attention_mask: Optional[torch.Tensor] = None,  # torch.Size(bsz, n_images, \sum_{l=0}^{L-1} H_l \cdot W_l) 表示图片信息的掩码，哪些图片的tokrn是padding得到的
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        
        # self.sampling_offsets = nn.Linear(self.by_pass_hidden_size, self.num_heads * self.num_levels * self.num_points * 2)
        _, num_images, _ = image_positions.size()
        bsz, num_queries, by_pass_hidden_size = by_pass_hidden_states.size()
        assert(len(self.spatial_shapes) == self.num_levels * num_images)
        spatial_shapes = torch.as_tensor(self.spatial_shapes, dtype=torch.long, device=hidden_states.device)
        assert(multi_scale_images_features.shape[2] == (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
        image_attention_mask_5d = self.image_expand_mask(image_attention_mask, by_pass_hidden_states.dtype)                 # [bsz, num_images, 1, 1, image_feature_length]    

        if by_pass_hidden_states.size() != (bsz, self.num_queries, self.by_pass_hidden_size):
            raise ValueError(
                f"`by_pass_hidden_states` should be of size {(bsz, self.num_queries, self.by_pass_hidden_size)}, but is"
                f" {by_pass_hidden_states.size()}"
            )
        attn_weights = []
        attn_output = [torch.zeros(bsz, num_images * self.num_queries, self.by_pass_hidden_size)]
        image_hidden_states_per_batch = []
        
        for image_index in range(num_images):
            image_hidden_states_per_batch = [hidden_states[b, image_positions[b, image_index, 0]: image_positions[b, image_index, 1]] for b in range(bsz)]
            image_hidden_states = torch.nn.utils.rnn.pad_sequence(image_hidden_states_per_batch, batch_first=True, padding_value=-100)    
            prompt_attention_mask = image_hidden_states.ne(-100)[..., 0]                                                          # (bsz, image_prompt_length)
            image_prompt_length = image_hidden_states.shape[1]
            prompt_attention_mask_4d = expand_mask(prompt_attention_mask, image_hidden_states.dtype, self.num_queries, self.num_levels * self.num_points)           # (bsz, 1, num_queriues + prompt_len, value_len)
            image_attention_mask_4d = image_attention_mask_5d[:, image_index]                                                                                       # [bsz, 1, 1, image_feature_length]    
              
            image_prompt = self.proj_to_prompt(image_hidden_states)                                                             # (bsz, image_prompt_length(coarse-grained-image-token-length), by_pass_hidden_size) 
            prompt_together_queries = torch.cat((image_prompt, by_pass_hidden_states), dim = 1)                                 # (bsz, image_prompt_length + self.num_queries, by_pass_hidden_size)
            reference_points, spatial_shapes, level_start_index = deform_inputs(by_pass_hidden_states, multi_scale_images_features, self.spatial_shapes)
            # reference_points.shape = torch.Size([1, image_prompt_length + num_queries, 1, 2])
            # spatial_points.shape = torch.Size([n_images * n_levels, 2])          default: n_levels=4
            # level_start_index = torch.Size([n_images * n_levels])
            reference_points = reference_points.repeat(bsz, 1, self.num_levels, 1)
            
            # 计算sampling_offsets:
            sampling_offsets = self.sampling_offsets(prompt_together_queries)                                                   # 等号右边： (bsz, num_queries + image_prompt_length, self.num_heads * self.num_levels * self.num_points * 2)
            sampling_offsets = sampling_offsets.view(bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points, 2)     # sampling_offsets.shape = torch.Size([bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points, 2])
            
            # 计算attention_weights:
            # self.attention_weights = nn.Linear(self.by_pass_hidden_size, self.num_heads * self.num_levels * self.num_points)
            attention_weights = self.attention_weights(prompt_together_queries).view(bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels * self.num_points)
            attention_weights = attention_weights + prompt_attention_mask_4d.transpose(1, 2) + image_attention_mask_4d.transpose(1, 2)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points)       
            # attention_weights的size是 (bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points)  
            if attention_weights.size() != (bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points):
                raise ValueError(
                    f"`attention_weights` should be of size {(bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points)}, but is"
                    f" {attention_weights.size()}"
                )
            attn_weights.append(attention_weights)
            
            
            # 计算value_states：
            # self.image_v_proj = nn.Linear(config.mm_hidden_size, self.by_pass_hidden_size)                                    
            value_states = self.image_v_proj(multi_scale_images_features[:, image_index])                                                 # (bsz, \sum_{l=0}^{L-1} H_l \cdot W_l, mm) 作为value
            _, num_values, _ = value_states.size()
            assert num_values == torch.sum(multi_scale_images_shape[0, image_index], dim = -1)
            value_states = value_states.view(bsz, num_values, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            # 计算spatial_shapes, level_start_index等等
            level_start_index_per_image = level_start_index[image_index * self.num_levels, (image_index + 1) * self.num_levels] 
            if level_start_index_per_image.size() != (self.num_levels, ):
                raise ValueError(
                    f"`level_start_index_per_image` should be of size {(self.num_levels, )}, but is"
                    f" {level_start_index_per_image.size()}"
                )
            spatial_shapes_per_image = spatial_shapes[image_index * self.num_levels, (image_index + 1) * self.num_levels]
            if spatial_shapes_per_image.size() != (self.num_levels, 2):
                raise ValueError(
                    f"`level_start_index_per_image` should be of size {(self.num_levels, 2)}, but is"
                    f" {spatial_shapes_per_image.size()}"
                )
                
            # sampling_offsets.shape = torch.Size([bsz, num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points, 2])
            # reference_points.shape = torch.Size([bsz, num_queries + image_prompt_length, self.num_levels, 2])
            sampling_locations = sampling_offsets + reference_points[:, :, None, :, None, :]                                                         
            if sampling_locations.size() != (bsz, self.num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points, 2):
                raise ValueError(
                    f"`level_start_index_per_image` should be of size {(bsz, self.num_queries + image_prompt_length, self.num_heads, self.num_levels, self.num_points, 2)}, but is"
                    f" {sampling_locations.size()}"
                )
                
            if value_states.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(                                                
                value_states, spatial_shapes_per_image, level_start_index_per_image, sampling_locations,
                attention_weights, self.im2col_step)
            if output.size() != (bsz, self.num_queries + image_prompt_length, by_pass_hidden_size):
                raise ValueError(
                    f"`output` should be of size {(bsz, self.num_queries + image_prompt_length, by_pass_hidden_size)}, but is"
                    f" {output.size()}"
                )
            attn_output[:, image_index * self.num_queries : (image_index + 1) * self.num_queries] = output[:, -self.num_queries]
        
        attn_output = self.cross_attn_1_o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value


class QuerySelfAttention(nn.Module):
    def __init__(self, config: LlavaFGloraConfig):
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.by_pass_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        self.self_attn_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.self_attn_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.self_attn_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.self_attn_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)
        
        self._init_rope()
        
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        by_pass_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, num_queries, _ = by_pass_hidden_states.size()
        
        query_states = self.self_attn_q_proj(by_pass_hidden_states).view(bsz, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.self_attn_k_proj(by_pass_hidden_states).view(bsz, num_queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.self_attn_v_proj(by_pass_hidden_states).view(bsz, num_queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, num_queries, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, num_queries, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, num_queries, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, num_queries, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, num_queries, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, num_queries, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, num_queries, self.hidden_size)
        attn_output = self.self_attn_o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class TextQueryCrossAttention(nn.Module):
    def __init__(self, config: LlavaFGloraConfig):
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        self.num_queries = config.num_queries
        self.by_pass_hidden_size = config.hidden_size
        
        self.cross_attn_2_q_proj = nn.Linear(self.by_pass_hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.cross_attn_2_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.cross_attn_2_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.cross_attn_2_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)
        
        self._init_rope()
        
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def forward(
        self,
        hidden_states: torch.Tensor,
        by_pass_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, seq_length, _ = hidden_states.size()
        bsz, num_queries, _ = by_pass_hidden_states.size()
        assert num_queries % self.num_queries == 0, f"num_queries ({num_queries}) must be divisible by config.num_queries ({self.num_queries})"

        query_states = self.cross_attn_2_q_proj(hidden_states).view(bsz, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.cross_attn_2_k_proj(by_pass_hidden_states).view(bsz, num_queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.cross_attn_2_v_proj(by_pass_hidden_states).view(bsz, num_queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)         # (bsz, self.num_heads, seq_length, self.head_dim) * (bsz, self.num_key_value_heads, self.head_dim, num_queries) = (bsz, self.num_heads, seq_length, num_queries)
        
        if attn_weights.size() != (bsz, self.num_heads, seq_length, num_queries):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_length, num_queries)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_length, num_queries):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_length, num_queries)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)          
        attn_output = torch.matmul(attn_weights, value_states)         # (bsz, self.num_heads, seq_length, num_queries) * (bsz, self.num_key_value_heads, num_queries, self.head_dim) = (bsz, self.num_heads, seq_length, self.head_dim)                                         
        
        if attn_output.size() != (bsz, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_length, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_length, self.hidden_size)
        attn_output = self.cross_attn_2_o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value
