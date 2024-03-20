import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from train import DataCollatorForSupervisedDataset

model_name_or_path = "/liymai24/sjtu/bokai/LLaVA/checkpoints/llava-v1.5-13b-pretrain"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
print("tokenizer.pad_token_id:", tokenizer.pad_token_id)

# 使用分词器对文本进行分词
text1 = "Hello, how are you?"
answer1 = "My name is LLava"
text2 = "Do you know what is love?"
answer2 = "I am not sure out what love is, but there is one thing for sure, that is, love is not related to you "
text3 = "What can I say? Mamba out!"
answer3 = "Man! You know I am saying!"

# Create lists to store the processed data
input_ids = []
labels = []

# Process each text-answer pair
for text, answer in zip([text1, text2, text3], [answer1, answer2, answer3]):
    # Tokenize the text and answer
    text_tokens = tokenizer.tokenize(text, add_special_tokens=True)  # Add special tokens (CLS, SEP)
    answer_tokens = tokenizer.tokenize(answer, add_special_tokens=False)  # Don't add special tokens

    # Convert tokens to their corresponding IDs
    input_ids.append(tokenizer.convert_tokens_to_ids(text_tokens))
    labels.append(tokenizer.convert_tokens_to_ids(answer_tokens))

# Print the processed data in the desired format
for i in range(3):
    print(f'input_ids{i+1}: {input_ids[i]}')
    print(f'labels{i+1}: {labels[i]}')

# tokenizer.pad_token_id: 0
# input_ids1: [1, 15043, 29892, 920, 526, 366, 29973]
# labels1: [1619, 1024, 338, 27624, 879]
# input_ids2: [1, 1938, 366, 1073, 825, 338, 5360, 29973]
# labels2: [306, 626, 451, 1854, 714, 825, 5360, 338, 29892, 541, 727, 338, 697, 2655, 363, 1854, 29892, 393, 338, 29892, 5360, 338, 451, 4475, 304, 366, 29871]
# input_ids3: [1, 1724, 508, 306, 1827, 29973, 341, 20027, 714, 29991]
# labels3: [2315, 29991, 887, 1073, 306, 626, 5934, 29991]
# Calculate attention mask

instances = [
    {"input_ids": torch.tensor([1, 15043, 29892, 920, 526, 366, 29973]), "labels": torch.tensor([1619, 1024, 338, 27624, 879])},
    {"input_ids": torch.tensor([1, 1938, 366, 1073, 825, 338, 5360, 29973]), "labels": torch.tensor([306, 626, 451, 1854, 714, 825, 5360, 338, 29892, 541, 727, 338, 697, 2655, 363, 1854, 29892, 393, 338, 29892, 5360, 338, 451, 4475, 304, 366, 29871])},
    {"input_ids": torch.tensor([1, 1724, 508, 306, 1827, 29973, 341, 20027, 714, 29991]), "labels": torch.tensor([2315, 29991, 887, 1073, 306, 626, 5934, 29991])},
]

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
batch = data_collator(instances)
print(instances)
print(batch)

# instances:
# [{'input_ids': tensor([    1, 15043, 29892,   920,   526,   366, 29973]), 'labels': tensor([ 1619,  1024,   338, 27624,   879])}, 
# {'input_ids': tensor([    1,  1938,   366,  1073,   825,   338,  5360, 29973]), 'labels': tensor([  306,   626,   451,  1854,   714,   825,  5360,   338, 29892,   541,
#           727,   338,   697,  2655,   363,  1854, 29892,   393,   338, 29892,
#          5360,   338,   451,  4475,   304,   366, 29871])}, 
# {'input_ids': tensor([    1,  1724,   508,   306,  1827, 29973,   341, 20027,   714, 29991]), 'labels': tensor([ 2315, 29991,   887,  1073,   306,   626,  5934, 29991])}]

# batch:
# {'input_ids': tensor([[    1, 15043, 29892,   920,   526,   366, 29973,     0,     0,     0],
#         [    1,  1938,   366,  1073,   825,   338,  5360, 29973,     0,     0],
#         [    1,  1724,   508,   306,  1827, 29973,   341, 20027,   714, 29991]]), 
# 'labels': tensor([[ 1619,  1024,   338, 27624,   879,  -100,  -100,  -100,  -100,  -100,
#           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
#           -100,  -100,  -100,  -100,  -100,  -100,  -100],
#         [  306,   626,   451,  1854,   714,   825,  5360,   338, 29892,   541,
#            727,   338,   697,  2655,   363,  1854, 29892,   393,   338, 29892,
#           5360,   338,   451,  4475,   304,   366, 29871],
#         [ 2315, 29991,   887,  1073,   306,   626,  5934, 29991,  -100,  -100,
#           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
#           -100,  -100,  -100,  -100,  -100,  -100,  -100]]), 
# 'attention_mask': tensor([[ True,  True,  True,  True,  True,  True,  True, False, False, False],
#         [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
#         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])}






# 输出: ['hello', ',', 'how', 'are', 'you', '?']

# 将分词后的文本转换为模型可接受的输入表示形式
# inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# print(inputs)

# decoded_text = tokenizer.decode(inputs.input_ids.squeeze(), skip_special_tokens=True)
# print(decoded_text)
# {'input_ids': tensor([[    1, 15043, 29892,   920,   526,   366, 29973]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
