# 用来打印输出bert从embedding到最终层数据表示的代码段。
基于https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch，欢迎大伙去star大佬的仓库


# 环境
python 3.7
pytorch 1.1
pytorch-pretrained-bert 0.6.2

# 预训练模型
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：

pytorch_model.bin
bert_config.json
vocab.txt
预训练模型下载地址：
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
来自这里
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

# code
# 准备阶段
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

content = '我很开心'
PAD, CLS = '[PAD]', '[CLS]'
tokenizer = BertTokenizer.from_pretrained('bert_pretrain')
bert = BertModel.from_pretrained('bert_pretrain')
bert.eval() #关闭dropout

pad_size = 32 #源码里面的seq_len最长为32
token = tokenizer.tokenize(content)

>>> token
['我', '很', '开', '心']

token = [CLS] + token
seq_len = len(token)
mask = []
token_ids = tokenizer.convert_tokens_to_ids(token)

>>> token_ids
[101, 2769, 2523, 2458, 2552]

mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
token_ids += ([0] * (pad_size - len(token))) #补0

>>> mask
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

>>> token_ids 
[101, 2769, 2523, 2458, 2552, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

token_ids = torch.LongTensor(token_ids).unsqueeze(0)
mask = torch.LongTensor(mask).unsqueeze(0)

# embeddings
<!-- bert结构可以直接shell里面输入bert，我打印的结果在bert-structure.pdf里面 -->

<!-- 第一层embeddings层，原文（BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding）包含三种embeddings -->

>>> bert.embeddings
BertEmbeddings(
  (word_embeddings): Embedding(21128, 768, padding_idx=0)
  (position_embeddings): Embedding(512, 768)
  (token_type_embeddings): Embedding(2, 768)
  (LayerNorm): BertLayerNorm()
  (dropout): Dropout(p=0.1, inplace=False)
)

>>> bert.embeddings(token_ids) #完整的embedding，三个emb的累加
tensor([[[ 0.0654,  0.0782, -0.2377,  ..., -0.0000, -0.2482, -0.1240],
         [ 0.5552, -0.3279,  1.1990,  ..., -0.0000,  0.2500, -0.5213],
         [-0.3806, -1.1439,  0.0000,  ..., -0.5998, -0.4624,  0.9751],
         ...,
         [ 1.0584,  0.1734, -0.3305,  ...,  2.2539,  0.1763,  0.1000],
         [ 1.1806,  0.0889, -0.5790,  ...,  2.2402,  0.0000,  0.6468],
         [ 1.2342,  0.2373, -0.3310,  ...,  2.6081, -0.0313,  0.4703]]])

>>> bert.embeddings.word_embeddings(token_ids) #word embedding 
tensor([[[ 0.0174, -0.0005, -0.0052,  ..., -0.0172, -0.0042,  0.0015],
         [ 0.0428, -0.0233,  0.0805,  ..., -0.0529,  0.0309, -0.0479],
         [-0.0270, -0.0754,  0.0501,  ..., -0.0359, -0.0453,  0.0391],
         ...,
         [ 0.0262,  0.0109, -0.0187,  ...,  0.0903,  0.0028,  0.0064],
         [ 0.0262,  0.0109, -0.0187,  ...,  0.0903,  0.0028,  0.0064],
         [ 0.0262,  0.0109, -0.0187,  ...,  0.0903,  0.0028,  0.0064]]])

<!-- 源码里面有一些中间部分计算过程可以go to definition查看
class BertEmbeddings() -->
seq_length = token_ids.size(1)
position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)

>>> position_ids
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

position_ids = position_ids.unsqueeze(0).expand_as(token_ids) # unsqueeze
token_type_ids = torch.zeros_like(token_ids)

>>> bert.embeddings.position_embeddings(position_ids) #position embedding
tensor([[[-0.0011,  0.0083, -0.0123,  ..., -0.0281, -0.0158, -0.0110],
         [-0.0063,  0.0018, -0.0067,  ..., -0.0717, -0.0118,  0.0171],
         [ 0.0069,  0.0052, -0.0182,  ..., -0.0569,  0.0204,  0.0215],
         ...,
         [ 0.0303, -0.0031,  0.0099,  ...,  0.0173,  0.0111,  0.0012],
         [ 0.0360, -0.0078, -0.0026,  ...,  0.0165,  0.0211,  0.0288],
         [ 0.0392,  0.0001,  0.0099,  ...,  0.0415,  0.0006,  0.0203]]])

>>> bert.embeddings.token_type_embeddings(token_type_ids) #segment embedding
tensor([[[ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003],
         [ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003],
         [ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003],
         ...,
         [ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003],
         [ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003],
         [ 0.0010,  0.0012, -0.0060,  ...,  0.0441, -0.0023, -0.0003]]])

embeddings = bert.embeddings.word_embeddings(token_ids).add(bert.embeddings.position_embeddings(position_ids)).add(bert.embeddings.token_type_embeddings(token_type_ids))

embeddings = bert.embeddings.LayerNorm(embeddings)
embeddings = bert.embeddings.dropout(embeddings)

>>> bert.embeddings(token_ids) == embeddings #正常是全部相等
tensor([[[True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         ...,
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True]]])

>>> bert.embeddings(token_ids) #模块化的embedding
tensor([[[ 0.0588,  0.0704, -0.2139,  ..., -0.0237, -0.2234, -0.1116],
         [ 0.4997, -0.2951,  1.0791,  ..., -0.9039,  0.2250, -0.4692],
         [-0.3425, -1.0295,  0.4302,  ..., -0.5398, -0.4162,  0.8776],
         ...,
         [ 0.9526,  0.1560, -0.2974,  ...,  2.0285,  0.1587,  0.0900],
         [ 1.0625,  0.0800, -0.5211,  ...,  2.0162,  0.3458,  0.5821],
         [ 1.1108,  0.2136, -0.2979,  ...,  2.3473, -0.0281,  0.4232]]],
       grad_fn=<AddBackward0>)
>>> embeddings # 单步调试的embedding
tensor([[[ 0.0588,  0.0704, -0.2139,  ..., -0.0237, -0.2234, -0.1116],
         [ 0.4997, -0.2951,  1.0791,  ..., -0.9039,  0.2250, -0.4692],
         [-0.3425, -1.0295,  0.4302,  ..., -0.5398, -0.4162,  0.8776],
         ...,
         [ 0.9526,  0.1560, -0.2974,  ...,  2.0285,  0.1587,  0.0900],
         [ 1.0625,  0.0800, -0.5211,  ...,  2.0162,  0.3458,  0.5821],
         [ 1.1108,  0.2136, -0.2979,  ...,  2.3473, -0.0281,  0.4232]]],
       grad_fn=<AddBackward0>)

#现在我们拿到了embedding层的输出。

# encoder
<!-- 然后是encoder部分，源码的class BertEncoder(nn.Module),源码里面用做分类任务的是多头注意力的最后一个结果，我们也用最后一个 -->
>>> bert.encoder.layer[-1]
BertLayer(
  (attention): BertAttention(
    (self): BertSelfAttention(
      (query): Linear(in_features=768, out_features=768, bias=True)
      (key): Linear(in_features=768, out_features=768, bias=True)
      (value): Linear(in_features=768, out_features=768, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (output): BertSelfOutput(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (LayerNorm): BertLayerNorm()
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (intermediate): BertIntermediate(
    (dense): Linear(in_features=768, out_features=3072, bias=True)
  )
  (output): BertOutput(
    (dense): Linear(in_features=3072, out_features=768, bias=True)
    (LayerNorm): BertLayerNorm()
    (dropout): Dropout(p=0.1, inplace=False)
  )
)

attention_mask = mask
input_ids = token_ids # 主要是保持和源码变量一致，避免错误加复制起来比较快
token_type_ids = torch.zeros_like(input_ids)


extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #2dmask转换成3d方便multi-heads广播
extended_attention_mask = extended_attention_mask.to(dtype=next(bert.parameters()).dtype) # fp16 compatibility
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

>>> extended_attention_mask
tensor([[[[    -0.,     -0.,     -0.,     -0.,     -0., -10000., -10000.,
           -10000., -10000., -10000., -10000., -10000., -10000., -10000.,
           -10000., -10000., -10000., -10000., -10000., -10000., -10000.,
           -10000., -10000., -10000., -10000., -10000., -10000., -10000.,
           -10000., -10000., -10000., -10000.]]]])

embedding_output = embeddings

all_encoder_layers = [] #12 heads
for layer_module in bert.encoder.layer:
    hidden_states = layer_module(embedding_output, extended_attention_mask)
    all_encoder_layers.append(hidden_states)


<!-- layer_module部分实现细节 -->

# attention
mixed_query_layer = bert.encoder.layer[11].attention.self.query(embeddings) # shape [1, 32, 768], layer[x] x->[0,11]
mixed_key_layer = bert.encoder.layer[11].attention.self.key(embeddings)
mixed_value_layer = bert.encoder.layer[11].attention.self.value(embeddings)


<!-- 模型里的细节，mself.all_head_size = self.num_attention_heads * self.attention_head_size
12heads，每个head生成的vector维度是64维，12*64=768
self.query = nn.Linear(config.hidden_size, self.all_head_size)
self.key = nn.Linear(config.hidden_size, self.all_head_size)
self.value = nn.Linear(config.hidden_size, self.all_head_size)
q，k，v的神经网络也是二维的，hidden_size(768) * all_head_size(12) -->


query_layer = bert.encoder.layer[11].attention.self.transpose_for_scores(mixed_query_layer) # size [1, 12, 32, 64]
key_layer = bert.encoder.layer[11].attention.self.transpose_for_scores(mixed_key_layer)
value_layer = bert.encoder.layer[11].attention.self.transpose_for_scores(mixed_value_layer)

#Take the dot product between "query" and "key" to get the raw attention scores.
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
attention_scores = attention_scores / math.sqrt(bert.encoder.layer[11].attention.self.attention_head_size)


#Apply the attention mask is (precomputed for all layers in BertModel forward() function)
attention_mask = mask.unsqueeze(1).unsqueeze(2)
attention_mask = attention_mask.to(dtype=next(bert.parameters()).dtype)
attention_scores = attention_scores + attention_mask

import torch.nn as nn
#Normalize the attention scores to probabilities.
attention_probs = nn.Softmax(dim=-1)(attention_scores)

#This is actually dropping out entire tokens to attend to, which might
#seem a bit unusual, but is taken from the original Transformer paper.
attention_probs = bert.encoder.layer[11].attention.self.dropout(attention_probs)

context_layer = torch.matmul(attention_probs, value_layer)
context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
new_context_layer_shape = context_layer.size()[:-2] + (bert.encoder.layer[11].attention.self.all_head_size,) #size [1, 32, 768]
context_layer = context_layer.view(*new_context_layer_shape)

self_output = context_layer

attention_output = bert.encoder.layer[11].attention.output(context_layer, embeddings) # size [1, 32, 768]


hidden_states = bert.encoder.layer[11].intermediate.dense(attention_output)
hidden_states = bert.encoder.layer[11].intermediate.intermediate_act_fn(hidden_states)
intermediate_output = hidden_states


hidden_states = bert.encoder.layer[11].output.dense(intermediate_output)
hidden_states = bert.encoder.layer[11].output.dropout(hidden_states)
layer_output = bert.encoder.layer[11].output.LayerNorm(hidden_states + attention_output)



sequence_output = all_encoder_layers[-1]
pooled_output = bert.pooler(sequence_output) #用来做序列分类
<!-- 算的结果不一样，但是debug代码确实是这么走的，不知道哪里出了问题 -->