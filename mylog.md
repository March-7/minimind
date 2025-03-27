# 记录训练kxgpt

复现自项目</url>https://github.com/jingyaogong/minimind

## 1、训练tokenizer
- 使用ByteLevel-bpe算法
- 词表大小6600
- 特殊token ["\<unk>", "\<s>", "\</s>"]
- 耗时约：30min

![tokenizer训练](./images/tokenizer-train.png)

## 2、预训练（dense，类似llama3架构）
- llama3特殊点： RoPE、swiglu激活函数、mlp中的门控机制
```
输入 x
│
├─> w1(x) ──> silu ─┐
│                   × ──> w2 ──> dropout ──> 输出
└─> w3(x) ─────────┘
```
```
模型架构超参数：
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        vocab_size: int = 6600,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 8192,
        rope_theta: int = 1e6,
        dropout: float = 0.0,
        flash_attn: bool = True,
```
- 训练了4个epoch
- 总参数量：25.932 百万(26M)
- 耗时：161min

## 3、分阶段sft
- 第一阶段Full SFT 数据的seq_length = 512  路径 ./minimind/dataset/sft_512.jsonl 
- epoch：1
- batchsize：32 耗时：192min
- 第一阶段Full SFT 数据的seq_length = 2048  路径 ./minimind/dataset/sft_2048.jsonl
- epoch：1
- batchsize: 32 耗时：13.8h

## 4、dpo
- Full RLHF 数据路径：./dataset/dpo.jsonl
- sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
- epoch：2
- batchsize：8 耗时：239min
- dpo的很大区别在于需要policy model和ref model，损失函数差别大，常用logsigmoid

## 5、知识蒸馏（只学习代码）
- 黑盒蒸馏与sft一样，知识数据从教师模型的输出收集
- 在SFT学习中，模型的目标是拟合词Token分类硬标签（hard labels），即真实的类别标签（如 0 或 6400）
- 白盒蒸馏中，教师模型的softmax概率分布被用作软标签（soft labels）。小模型仅学习软标签，并使用KL-Loss来优化模型的参数。【−P(i)logQ(i)】

## 6、LoRA 
- 为基础模型增加了LoRA外挂，这个过程并不损失基础模型的本身能力，且参数训练量极小
```
LLM 总参数量: 26194432
LoRA 参数量: 262144
LoRA 参数占比: 1.00%
```
- 训练过程和sft一样
- 自我认知，耗时：0min
- 医疗垂类，耗时：52min

## 7、Reason ditillation
- 蒸馏和SFT一样，但实验结果是模型难以每次都符合模板规范的回复，即脱离思考和回复标签约束。 这里的小技巧是增加标记位置token的损失惩罚
```
# 在 sp_ids 对应的位置增加额外的惩罚
...
loss_mask[sp_ids] = 10 # 惩罚系数,原本是1
```
- epoch：1
- 耗时：13.6min