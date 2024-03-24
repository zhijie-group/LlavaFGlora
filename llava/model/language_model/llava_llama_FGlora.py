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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import warnings
import einops
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .by_pass_attention import TextQformerCrossAttention, QueryImageCrossAttention, QuerySelfAttention, TextQueryCrossAttention
from ..config import LlavaFGloraConfig, PerceiverConfig
from ...model.multimodal_encoder.visual_tokenizer import VisualTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

logger = logging.get_logger(__name__)

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

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    
    # torch.finfo(dtype).min表示给定dtype的最小正子正规化浮点数，用于初始化掩码
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    # 使用masked_fill函数，将掩码中的1替换为指定的填充值，以确保在后续的计算中被忽略
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlavaFGloraAttention(LlamaAttention):
    def __init__(self, config: LlavaFGloraConfig):
        super().__init__(config)
        self.config = config
        self.query_text_cross_attn = TextQformerCrossAttention(config)
        self.query_image_cross_attn = QueryImageCrossAttention(config)
        self.query_self_attn = QuerySelfAttention(config) if config.if_query_self_attn else None
        self.text_query_cross_attn = TextQueryCrossAttention(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        multi_scale_images_features: torch.Tensor,                      # torch.size([bsz, n_images, \sum_{l=1}^{l=L} h_l * w_l, image_embed_dim])
        multi_scale_images_shape: List[Tuple[int, int]],                         # e.g. [(12, 12), (24, 24), (48, 48), (96, 96)]
        image_positions: torch.Tensor,                                  # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: torch.Tensor = None,                      # 就像 hiddens_states里面有对应的attention_mask，知道哪些部分的token是padding得到的, multi_scale_images_features也有padding得到的部分, 对应的image_attnetion_mask  
                                                                        # torch.Size(bsz, n_images, \sum_{l=0}^{L-1} H_l \cdot W_l) 
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
        
        # 获取by_pass_hidden_states                         (bsz, self.num_queries, embed_dim)
        by_pass_hidden_states, _, _ = self.query_text_cross_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # 讲高分辨率图片信息融入
        by_pass_hidden_states, _, _ = self.query_image_cross_attn(
            hidden_states = hidden_states,
            by_pass_hidden_states = by_pass_hidden_states,
            multi_scale_images_features = multi_scale_images_features,
            spatial_shapes = multi_scale_images_shape,
            image_positions = image_positions,
            image_attention_mask = image_attention_mask,
            position_ids = position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
                
        if self.config.if_query_self_attn:
            by_pass_hidden_states, _, _ = self.query_self_attn(
                by_pass_hidden_states=by_pass_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        
        hidden_states, self_attn_weights, past_key_value = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        residual = hidden_states
        
        hidden_states, _, _ = self.text_query_cross_attn(
            hidden_states=hidden_states,
            by_pass_hidden_states = by_pass_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states += residual
        
        return hidden_states, self_attn_weights, past_key_value

class LlavaFGloraDecoderLayer(nn.Module):
    def __init__(self, config: LlavaFGloraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlavaFGloraAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        multi_scale_images_features: torch.Tensor,                      # torch.size([bsz, n_images, \sum_{l=1}^{l=L} h_l * w_l, image_embed_dim])
        multi_scale_images_shape: List[Tuple[int, int]],                # e.g. [(12, 12), (24, 24), (48, 48), (96, 96)]
        image_positions: torch.Tensor,                                  # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: torch.Tensor = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            multi_scale_images_features=multi_scale_images_features,
            multi_scale_images_shape=multi_scale_images_shape,
            image_positions=image_positions,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class LlavaFGloraModel(LlamaModel, LlavaMetaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlavaFGloraConfig):
        super().__init__(config)
        self.config= config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)       # 32000 4096 0
        self.layers = nn.ModuleList([LlavaFGloraDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def initialize_extract_multi_feature_modules(self, model_args, fsdp=None):
        self.visual_tokenizer = VisualTokenizer(vision_tower=self.vision_tower.vision_tower,        # self.vision_tower.vision_tower 对应的class CLIPVisionModel
                                            perceiver_config=model_args.perceiver_config,
                                            llm_hidden_size=model_args.hidden_size)
        # self.visual_tokenizer = VisualTokenizer(encoder_model_path=model_args.image_feature_extractor,
        #                                     pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        #                                     perceiver_config=model_args.perceiver_config,
        #                                     llm_hidden_size=model_args.hidden_size)
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        multi_scale_images_features: torch.Tensor,                      # torch.size([bsz, n_images, \sum_{l=1}^{l=L} h_l * w_l, image_embed_dim])
        multi_scale_images_shape: List[Tuple[int, int]],                # e.g. [(12, 12), (24, 24), (48, 48), (96, 96)]
        image_positions: torch.Tensor,                                  # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: torch.Tensor = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    multi_scale_images_features,
                    multi_scale_images_shape,
                    image_positions,
                    attention_mask,
                    image_attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    multi_scale_images_features=multi_scale_images_features,                        # torch.size([bsz, n_images, \sum_{l=1}^{l=L} h_l * w_l, image_embed_dim])
                    multi_scale_images_shape=multi_scale_images_shape,                           # e.g. [(12, 12), (24, 24), (48, 48), (96, 96)]
                    image_positions=image_positions,                                    # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
                    attention_mask=attention_mask,
                    image_attention_mask=image_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlavaLlamaFGloraForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaFGloraConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaFGloraConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.config = config
        self.model = LlavaFGloraModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_model(self):
        return self.model
    
    def prepare_inputs_images_features(
        self,
        input_ids: torch.LongTensor,
        image_tensors: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
    ):
        # step 1. get text token embeds

        assert num_image_per_seq.sum() == image_tensors.shape[0], (
            f"image_tensors.shape: {image_tensors.shape} | "
            f"num_image_per_seq.sum(): {num_image_per_seq.sum()}"
        )

        # step 2. get image embeds
        output = self.model.visual_tokenizer(image_tensors)


        # step 4. prepare cross attention mask and MMFS features for mm decoder
        output.update(
            self._prepare_mmfs_features_for_mm_decoder(
                input_ids,
                num_image_per_seq,
                output["multiscale_features"],
            )
        )

        return output

    def _prepare_mmfs_features_for_mm_decoder(
        self,
        input_ids: torch.LongTensor,
        num_image_per_seq: Optional[torch.Tensor] = None,           # 不同样本里面不同的照片数量
        multiscale_features=None,
    ):
        output = {}

        B, L = input_ids.shape

        max_num_image = int(num_image_per_seq.max())
        
        # 通过设置 as_tuple=True 参数，.nonzero() 方法将返回一个元组，其中包含两个张量，分别表示非零元素的行索引和列索引
        # mask = torch.tensor([[0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0]])
        # soi_token_pos = mask.nonzero(as_tuple=True)
        # print(soi_token_pos)   (tensor([0, 0, 1, 1, 1]), tensor([3, 5, 1, 3, 5]))
        # 行索引是bsz的位置，而列索引才是在seq_length维度的位置
        soi_token_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1]
        image_token_pos = -1 * torch.ones(B, max_num_image).type_as(soi_token_pos)
        start_idx = 0
        
        # tensor([3, 5, 1, 3, 5]) -> tensor([[3, 5, -1], [1, 3, 5]])
        for i in range(B):          # Start of Image i.e. soi
            image_token_pos[i, : int(num_image_per_seq[i])] = (
                soi_token_pos[start_idx : start_idx + int(num_image_per_seq[i])] + 1
            )
            start_idx = start_idx + int(num_image_per_seq[i])



# ***************************************************************************************************************
        mmfs_features = []                      
        for feat in multiscale_features:        #       torch.Size([8, 1024, 96, 96])  torch.Size([8, 1024, 48, 48])    
                                                #       torch.Size([8, 1024, 24, 24])  torch.Size([8, 1024, 12, 12])
            mmfs_features.append(feat)
        mmfs_features_new = [
            torch.zeros(
                B,  
                max_num_image,
                *feat.shape[1:],
                device=feat.device,
                dtype=feat.dtype,
            )
            for feat in mmfs_features
        ]
        for feat, feat_n in zip(mmfs_features, mmfs_features_new):
            start_idx = 0
            for i in range(B):
                item = feat[start_idx : start_idx + int(num_image_per_seq[i])]
                feat_n[i, : item.shape[0], ...] = item
                start_idx = start_idx + int(num_image_per_seq[i])

        mmfs_features_mm = []
        for feat in mmfs_features_new:
            feat_n = einops.rearrange(feat, "b n c h w -> b n (h w) c")
            mmfs_features_mm.append(feat_n)
        mmfs_features_mm = torch.cat(mmfs_features_mm, dim=2)
        output["mmfs_features_mm"] = mmfs_features_mm

        return output
    
    
    def forward(
        self,
        images: Optional[torch.FloatTensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                num_image_per_seq,
                image_positions                                 # torch.Size([bsz, n_images, 2])     表示n_images图片的起始和结束位置
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
            
   
            output = self.prepare_inputs_images_features(
                input_ids,
                image_tensors=images,
                num_image_per_seq=num_image_per_seq,
            )
            
            multi_scale_images_features = output.pop("mmfs_features_mm", None)
            multi_scale_images_shape = output.pop("spatial_shapes", None)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        
        image_attention_mask = torch.zeros(input_ids.shape[0], int(num_image_per_seq.max()))
        for i, num_images in enumerate(num_image_per_seq):
            image_attention_mask[i, :int(num_images)] = 1
        outputs = self.model(
            multi_scale_images_features=multi_scale_images_features,
            multi_scale_images_shape=multi_scale_images_shape,
            image_positions=image_positions,
            input_ids=None,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # print("logits:", logits[0, -10:, :10], logits.shape)
            # print("lables:", labels[0, :10], labels.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    
AutoConfig.register("llava_llama_FGlora", LlavaLlamaFGloraForCausalLM)           
AutoModelForCausalLM.register(LlavaFGloraConfig, LlavaLlamaFGloraForCausalLM)       # 通过注册自动生成的模型，我们可以使用AutoModelForCausalLM.from_pretrained("llava_llama")来加载和使用与"llava_llama"配置相关联的模型
