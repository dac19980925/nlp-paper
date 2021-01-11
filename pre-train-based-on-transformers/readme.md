transformer:Attention Is All You Need
解决的问题：机器翻译RNN模式的encoder和decoder模型不支持并行
提出的方法：使用self-attention来处理文本内部的依赖关系，相比于rnn，所有的运算都不依赖之前的隐藏状态，天然支持并行
结论：效果优于传统基于cnn，rnn的nlp模型，且训练更快
细节问题：Scaled Dot-Product Attention？
    文中提到了两个比较常用的注意力函数，additive attention、 dot-product (multi-plicative) attention。其中additive attention是两个向量的拼接经过前馈神经网络，sigmoid函数计算两者的相关度，也就是p=sigmoid（wx+b），其中x=x1+x2，一次只算一对q和k。dot-product attention使用矩阵运算，也就是文中的方法。计算速度想比additive attention更快，节省空间，主要是矩阵运算的优势。但是dk也就是k的维度较大时，点积的值也会很大，softmax的结果梯度会很小，所以会使用根号dk做缩放。

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（基于英文单词，非中文的词语）
解决的问题：bert设计是为了从无标签数据的上下文学习预训练的深层双向语言表示。MLM灵感来源于一篇1953年的论文！！！克服了gpt这种只能看到单向文本的问题。预训练语言表示降低了为不同nlp任务设定不同方案的需要。
提出的方法：在无标签语料上学习文本的表示。使用MLM，随机mask部分词汇，让模型预测每个位置应该正确的值，mask的词在开始不能被看到，让模型通过上下文填空，相当于理解了句子的意思。同时由于微调阶段没有随机mask，所以mask的词里面一部分会随机修改成其他的词，或者保持不变，但大部分还是会遮蔽为[mask]，主要是为了解决预训练微调不匹配以及泛化性能（改成其他的词）。NSP用来捕捉句子之间的关系，给定句子a和b，模型会判断说b是否是a的下一句，数据就是一般的a和b时连续的句子，另一半是从corpus随机抽的。用来handle机器问答，自然演绎推理这种依赖句子之间关系的任务。
结果：基于transforer的预训练微调模型，self-attention已经被证明在nlp领域很强，加上MLM，NSP这两个预训练任务，在各种下游自然语言处理任务上都效果很好，MLM和NSP算是论文的关键点。

ERNIE：Enhanced Representation through Knowledge Integration
解决的问题：bert基于字或wordpiece的单词mask，不利于提取实体或短语蕴含的信息
提出的方法：实体mask，短语mask，在bert的基础上，如果某个unit（中文单字或者wordpiece的单词）被mask，会加入一个匹配机制，将被mask的unit所属的短语，实体一起mask掉，最细的粒度仍是bert版本，相当于原来  “马斯克很厉害。”   原bert会mask 马，那么ERNIE会mask掉 马，斯， 克三个unit，短语也是同理，主要是为了让模型学到实体或短语背后表达的信息。同时预训练的DLM（Dialogue Language Model）任务和多领域数据做pretrain也提升了模型的效果。
结果：提升了效果，特别是在实体名称完形填空的任务上效果显著，这也很好解释，毕竟bert预训练预测的是随机mask的词，通常不连续，而ERNIE预训练任务就是猜的词或者实体名称，或者短语。一个bert的延申。
DLM是MLM的晋级版本，bert输入的句子对A和B会构造成对话中的上一句和下一句接茬的话。让模型预测query（A句）和response（B句）中mask的词。除此之外还会随机替换A和B中的某些词作为fake的样本，让模型判别real or fake。


bert-wwm:Pre-Training with Whole Word Masking for Chinese BERT
解决的问题：bert原文用的wordpiece将长单词分成组合词，