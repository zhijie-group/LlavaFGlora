# LlavaFGlora
 # 关于模型的另一个分路：
 处理高分辨率图像的，先是生成query，query-image-cross-attn，query-hidden_states-cross-attn，在LLaVA/llava/model/language_model/by_pass_attention.py位置，先只把save_mem=True跑通了；
 # attention:
 关于将三个模块：生成query，query-image-cross-attn，query-hidden_states-cross-attn，组合一起，重写了一个attention的位置在LLaVA/llava/model/language_model/llava_llama_FGlora.py里面，将attention->layer->model也在这个文件里面。

 目前不知道三个模块之间是否需要加点layernorm()之类的？
 
 文件里面最终的模型是LlavaLlamaFGloraForCausalLM，self.prepare_inputs_labels_for_multimodal是用于将文字+图片变成imput_embeds的，self.prepare_inputs_images_features是获取图片的几个不同分辨率的feature_map的，大致就是特征图=卷积下采样+从VIT层中获取信息；目前代码还有些地方没改进，包括一张图片过了2次VIT，有些参数没传进去需要用default等。
 # 训练的代码在train.py里面。

 # 训练开销：
 目前微调，可训练的参数大概是7亿，也就是7b的十分之一，占存32G/40G左右，时间大约是9-10(gpu*day)，也就是一张卡上200h左右
