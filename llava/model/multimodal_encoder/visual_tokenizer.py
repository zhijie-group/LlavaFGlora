import torch
import torch.nn as nn

from einops import rearrange

from .vit_adapter import clip_vit_adapter_hf
from .vit_adapter.pos_embed import get_abs_pos, get_2d_sincos_pos_embed
from transformers import Blip2QFormerModel, Blip2QFormerConfig

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_queries=32,
        hidden_size=768,
        qk_normalization=False,
        gradient_checkpointing=True,
        **kwargs
    ) -> None:
        super().__init__()

        config = Blip2QFormerConfig(hidden_size=hidden_size, **kwargs)
        config.qk_normalization = qk_normalization
        self.blip2qformer = Blip2QFormerModel(config)

        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_size))
        self.queries.data.normal_(0, config.initializer_range)
        if gradient_checkpointing:
            self.blip2qformer.gradient_checkpointing_enable()

    def forward(self, **kwargs):
        query_embeds = kwargs.pop("query_embeds", self.queries)

        return self.blip2qformer(query_embeds=query_embeds, **kwargs)


class VisualTokenizer(nn.Module):
    def __init__(
        self,
        encoder_model_path="/liymai24/sjtu/bokai/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336",
        pretrain_mm_mlp_adapter="checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin",
        perceiver_config=None,
        llm_hidden_size=5120,
        clip_normalize=True,
        grid_size=16,
    ) -> None:
        super().__init__()
        self.clip_normalize = clip_normalize
        self.encoder = clip_vit_adapter_hf(model_path=encoder_model_path)
        encoder_hidden_size = perceiver_config.encoder_hidden_size

        self.pos_proj = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.pos_ln = nn.LayerNorm(encoder_hidden_size, eps=1e-6)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(encoder_hidden_size, grid_size, cls_token=True)
            ).float()
        ).requires_grad_(False)

        self.perceiver_resampler = PerceiverResampler(**perceiver_config)
        self.length = perceiver_config.num_queries
        self.post_ln = nn.LayerNorm(encoder_hidden_size, eps=1e-6)
        self.proj = nn.Linear(perceiver_config.hidden_size, llm_hidden_size)
        
        weights = torch.load(pretrain_mm_mlp_adapter)

        # 将加载的权重赋值给 self.proj 的权重参数
        self.proj.weight.data = weights["proj.weight"]
        self.proj.bias.data = weights["proj.bias"]


        if self.clip_normalize:
            # normalize image
            CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [
                0.26862954,
                0.26130258,
                0.27577711,
            ]
            mean, std = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
            mean, std = rearrange(mean, "c -> 1 c 1 1"), rearrange(std, "c -> 1 c 1 1")
            self.register_buffer("clip_mean", mean)
            self.register_buffer("clip_std", std)

    def print_parameters_stats(self, prefix=""):
        for name, module in self.named_children():
            print(
                f"# {prefix}{name} Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M"
            )
            print(
                f"# {prefix}{name} Trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.2f}M"
            )

    def forward(self, image):
        if self.clip_normalize:
            # normalize image
            image = (image - self.clip_mean) / self.clip_std

        model_output = self.encoder(image)
        image_embed = model_output.last_hidden_state
        multiscale_features = model_output.hidden_states

        multiscale_features_n = []
        for ms_feat in multiscale_features:
            pos_embed = get_abs_pos(
                self.pos_embed[1:], ms_feat.size(2) * ms_feat.size(3)
            )
            pos_embed = rearrange(pos_embed, "(h w) c -> c h w", h=ms_feat.size(2))
            ms_feat = ms_feat + pos_embed
            multiscale_features_n.append(ms_feat)
        multiscale_features = multiscale_features_n

        pos_embed = get_abs_pos(self.pos_embed, image_embed.size(1))
        qformer_inputs = self.pos_ln(self.pos_proj(image_embed))
        qformer_inputs = qformer_inputs + pos_embed
        image_embed = image_embed + pos_embed

        qformer_inputs = self.post_ln(qformer_inputs)
        vis_embed = self.perceiver_resampler(
            encoder_hidden_states=qformer_inputs,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]
        vis_embed = self.proj(vis_embed)

        output = dict(vis_embed=vis_embed)
        output["image_embeds"] = image_embed[:, 1:, :]                   # remove cls token
        output["multiscale_features"] = multiscale_features

        return output

