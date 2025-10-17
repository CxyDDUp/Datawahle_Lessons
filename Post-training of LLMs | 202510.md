此内容为自己在Datawahle平台所学习课程(Post-training of LLMs )所记载笔记📒

万分感谢开源作者🙏🎆🎆

# 第一章 课程介绍

## 1.1 课程介绍

### 1.1.1 大模型训练的概述

+ 两个阶段：

  + 预训练阶段：

    + 模型学习预测下一个词或者标记。

    + > ==我认为==，这是后训练阶段的基石模型，通常是由各大厂商或者学术机构研究出来的基础模型。由于其庞大的数据规模，需要很大的资源成本，所以这并非是我们平时涉及的部分，只需要学会选取合适领域的基座模型即可。
      >
      > <font color=red>但值得留意得是：</font>一些常见大规模模型提出的结构及方法，需要了解及深思，例如Transformer架构、多头潜在注意力（MLA）、混合专家（MoE）等

  + 后训练阶段：

    + 模型通过进一步训练以执行更具体的任务。

    + > ==我认为==，此阶段应该就是平时的横向课题任务，所涉及的对大模型应用开发的垂直落地，由于垂直领域数据集的优先呢

### 1.1.2 后训练方法概述

+ 监督微调（SFT）：
  + 带<font color=red>标注</font>的提示🔔-响应对应的训练模型
  + ==核心==：让模型模仿输入提示与输出响应间的映射关系
  + 适用范围：引入**新**行为或对模型进行**重大调整**
+ 直接偏好优化（DPO）：
  + 向模型展示同一提示下的“优质”和“劣质”答案，驱动模型学习（<font color=red>？？</font>Maybe类似于强化学习的奖励机制❌）
    + `DPO`通过构造性损失函数，让模型趋近优质响应而远离劣质响应。
    + Example：
      + “我是你的助手”：劣质响应
      + “我是您的AI助手”：优质响应
      + 使用`DPO`调整一个`Qwen`指令模型的“身份🆔认知”
+ 在线强化学习（Online RL）：
  + 让模型接收提示并==生成响应==，根据奖励函数对响应质量进行**评分**，模型再根据评分结果进行更新。
  + 如何获取奖励函数？？？
    + 人工对响应质量进行评判，即训练出一个与人类判断水准一致的评分函数。
      + 例如：近端策略优化
    + 可验证奖励
      + 适用范围：数学或编程等具有客观正确性标准的任务
      + 例如：数学验证器或单元测试来判定解题步骤/代码是否正确
  + Summry：
    + 利用正确性来作为奖励函数
    + 例子🌰：DeepSeek团队提出的`GRPO`算法

### 1.1.3 总结

1. 大模型训练一般是分为俩steps：预训练阶段（一般不考虑，直接调用）+后训练阶段（重点学习）
2. 常见的后训练技术又分为三种：SFT、DPO、Online RL
3. 记录一下关于DPO与Online RL此刻感受的区别所在

|         方面         |                             DPO                              |                          Online RL                           |
| :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       核心思想       | 将偏好数据（==人类标注==的优劣回答组合）直接建模为一个分类问题，通过最大偏好回答的隐式奖励来优化模型 | 使用强化学习框架：模型生成回答，奖励模型(RM)给出分数，通过策略梯度更新模型。 |
| 是否需要显式奖励模型 | ❌<br />因为DPO通过数学变换的方式，将奖励函数“隐式”地嵌入到损失函数中 |          ✅<br />必须先训练一个独立的奖励模型来打分           |
|   是否需要参考模型   |       ✅<br />用于计算KL散度，防止模型偏离原始行为太多        |                         ✅<br />同左                          |

<font color=red>Tips:</font>

> KL散度：即相对熵。
>
> + 衡量两个概率分布之间差异的重要工具
> + KL散度越大，则两个真实分布所差就越远！！！

## 1.2 后训练技术介绍

### 1.2.1 后训练

+ 从随机初始化的模型开始预训练
  + 从各类数据源学习知识，==通常数据量非常庞大==
  + **Result:** Get一个基础模型`base`
+ 从`base`模型开始，进行后训练
  + Target: 从精心筛选的数据中学习到响应模式。==垂直领域的应用==
  + **Result:** Update为一个指令模型或者对话模型
+ 还可以进一步进行后训练
  + Target: 调整模型行为或者增强特定能力。==垂直领域的细节应用==

### 1.2.2 预训练

+ 通常被视作：<font color=red>无监督学习</font>
+ 数据集特点：
  + 大规模
  + 无标注
  + 文本语料
+ 概率公式：

$p(x_{1:L})=p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)....p(x_L|x_{1:L-1})=\prod_{i=1}^{L}p(x_i|x_{1:L-1})$

+ 目的：使得模型被训练成能根据已经给出的标记==预测==下一个标记

### 1.2.3 后训练技术

+ 监督微调（SFT）：
  + 监督学习，即带标签的数据集
  + 提示：给模型的指令；响应：模型应该给出的理想回答
  + 此过程需要1000至10亿个标记，远少于预训练规模
  + <font color=red>训练损失关键区别:</font>仅对响应标记进行训练，而不涉及提示标记
+ 直接偏好优化（DPO）：
  + 创建包含提示及其对应**优质/劣质**响应的数据集。
  + 特点：针对任意一个提示🔔，可生成多个响应并筛选出优质与劣质样本
  + 目的：使得模型远离劣质响应并且学习优质响应
  + 此过程需要1000至10亿个标记，远少于预训练规模
+ 在线强化学习（Online RL）：
  + 只需提示集+奖励函数
  + 提示———>语言模型———>生成响应———>通过奖励函数对该响应进行评分———>利用评分更新模型
  + 需要1000至1000万个提示
  + 目的：通过模型自身生成的响应来最大化奖励值
+ 后训练的三个关键因素：
  + ==数据与算法的协同设计==：选择适宜的数据结构以及良好的协同设计
  + ==可靠的高效的算法库==：`HuggingFace TRL`+`Open RLLHF`+`veRL`+`Nemo RL`
  + ==合理的评估体系==：
    + 替代人类评判的`LLM`评估：`AlpacaEval` \ `MT Bench` \ `Arena Hard`
      + `AlpacaEval`: 给出一个指令让模型A和B回答，再由LLM判断胜负
      + `MT Bench`：(Multi-Turn Benchmark) 使用``LLM`生成80个复杂的多轮对话，模型需要连续回答每一轮。LLM对其进行打分
      + `Arena Hard`：从用户真实提交的“最难问题”中筛选出一批最具挑战性的指令，让``LLM`作为裁判
    + 指令模型静态基准：`LiveCodeBench`(热门代码基准) \ `AIME 2024/2025`
      + `LiveCodeBench`:：模型生成代码->在真实测试用例上运行->计算通过率
      + `AIME 2024/2025`：模型生成解题过程->判断最终答案是否正确
    + 知识与推理数据集：`GPQA` \ `MMLU Pro`
      + `GPQA`：评估模型在博士级别专业知识上的表现
      + `MMLU Pro`：测试跨学科、多任务理解能力
    + 指令遵循评估: `IFEval`
      + `IFEval`：对细粒度指令对遵循程度
    + 函数调用与智能体评估：`BFCL` \ `NexusBench` \ `TauBench` \ `ToolSandbox`
      + `BFCL`：给定用户请求，判断是否需要调用函数
      + `NexusBench`：多步骤任务的表现
      + `TauBench`：多工具协同和动态决策
      + `ToolSandbox`：用于测试模型调用真实工具的能力
    + <font color=red>值得注意⚠️的是</font>：
      + 提升单一指标相对容易
      + 重要前提：不损害其他领域的能力

# 第二章 监督微调理论及实践

## 2.1 语言模型的监督式微调(SFT)

### 2.1.1 概述

+ 监督式微调（Supervised Fine-Tuning, SFT）

  + 把通用语言模型转换成任务型助手的方法

+ 方法：提供训练==提示==与理想==响应==的成对数据，使得模型学会模仿示例中的回答

+ <font color=red>核心</font>：

  + 目的：让基础模型（只能预测下一个token）学会生成生成预期的回答
  + Steps：
    + **基础模型**：未经微调的LLM往往会给出泛泛或者重复的回应
    + **带标签的数据集**：收集并整理提示与理想响应的成对数据
    + **SFT训练：**对数据集进行微调，通过最小化响应的交叉熵损失来训练模型：
      + $$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^N \log \bigl(p_\theta(\text{Response}(i)\mid \text{Prompt}(i))\bigr)$$
      + Target：使得模型在每个提示🔔条件下生成理想目标响应的概率==最大化==，即最大化在提示条件下回应中所有 token 的联合概率

+ ==拓展知识点==（交叉熵损失函数）：

  + 基本任务形式：`(输入文本 x, 目标输出文本 y)`，且要求给定的输入`x`，生成与`y`尽可能接近的输出

  + 模型如何生成文本？ **答**：自回归模型，即从左到右逐个生成 token

    + 假设目标输出 `y` 是一个 token 序列：

      + $y = (y_1, y_2, \dots, y_T), \quad y_t \in \text{Vocabulary}$

    + 模型在生成第 `t` 个 token 时，会基于前面的上下文（包括输入$x$和已生成的 $y_1$ 到 $y_{t-1}$）来预测下一个 token 的概率分布：

      + $P(y_t \mid x, y_1,y_2...y_{t-1}; \theta)$

      + <font color=red>注意！！！</font>

        + > 1. 这里的$x$实则是一个输入序列
          > 2. 并且在模型后续都不改变这个$x$，仅仅是添加前文预测结合起来，作为模型的输入

    + 其中，$ \theta$为模型参数

  + 目标🎯：我们希望模型预测出的 token 分布，**尽可能接近真实的目标 token**。

    + 简言之，当真实目标为$y_t$的时候，我们希望模型在$y_t$这个token上的预测概率**尽可能高**

  + 最大似然估计**（MLE）**函数：

    + 我们希望最大化模型生成真实序列$y$的概率：
      + $P(y \mid x; \theta) = \prod_{t=1}^T P(y_t \mid x,y_1,y_2...y_{t-1}; \theta)$
    + 取对数，得到对数似然：
      + $\log P(y \mid x; \theta) = \sum_{t=1}^T \log P(y_t \mid x,y_1,y_2...y_{t-1}; \theta)$
    + 为了优化，我们通常**最小化负对数似然（Negative Log-Likelihood, NLL）**：
      + $\mathcal{L}(\theta) = -\log P(y \mid x; \theta) = -\sum_{t=1}^T \log P(y_t \mid x,  y_1,y_2...y_{t-1}; \theta)$

  + **负对数似然** = **交叉熵损失**

    + 定义：

      + 真实分布$p$：在正确token$y_t$处为1，其余为0（独热编码）
      + 模型预测分布$q$：
        + 是模型输出的 **`softmax` 概率分布**
        + $q_t = p(y_t|x,y_1,y_2...y_{t};\theta)$
        + 例如：模型认为第 1234 个词的概率是 0.7，其他词加起来是 0.3

    + 交叉熵定义为：

      + 用分布 $q$ 来编码来自分布 $q$的数据时所需的平均比特数

      + 数学定义： $H(p_t, q_t)= -\sum_{i} p_t(i) \log q_t(i)$

      + 由于真实分布独热编码的存在，交叉熵等价于

        + $H(p_t, q_t)= -\log P(y_t \mid x,  y_1,y_2...y_{t-1}; \theta)$

      + 所以，**单个 token 的负对数似然 = 真实分布与预测分布之间的交叉熵**

    + 因此，总损失 = 序列级交叉熵，整个损失函数为：

      + $\mathcal{L}(\theta) = \sum_{t=1}^T H(p_t, q_t)$
      + $$ \boxed{\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(y_t \mid x, y_1, \dots, y_{t-1}; \theta)} $$

### 2.1.2 SFT 的最佳使用场景

+ 激发新的模型行为：

  + 将预训练模型转变为能遵循特定指令的助理（==垂直领域的微调==）
  + 让不具备推理能力的模型学会推理
  + 让模型在没有明确说明的情况下使用特定工具

+ 提升模型能力

  + 利用强大的大模型生成高质量合成数据，通过训练把这些能力“蒸馏”到小模型中（<font color=red>未解？？</font>）

    + “知识蒸馏” = **合成数据生成** + **模型能力迁移**
    + 把大模型生成的合成数据去训练一个**小模型**
    + 小模型在学习这些数据的过程中，**间接学会了大模型的思维方式和表达能力**
    + 这个过程就像把大模型的“知识”或“智能”**浓缩（蒸馏）** 到一个小容器里，所以叫 **知识蒸馏（Knowledge Distillation）**

    🧠 类比：就像一位大学教授（大模型）写了很多标准答案，然后让一个高中生（小模型）通过做题和模仿这些答案来提升水平

    + <FONT COLOR= RED>最好混合使用合成 + 真实数据</FONT>
    + 操作清单：
      + 明确任务（对话/代码）
      + 收集/编写500+个prompt
      + 用大模型进行response（加上system prompt）
      + 清洗数据（去重、过滤、打分）
      + 构建训练数据集（instruction + response）
      + 用 LoRA 微调小模型（如 Qwen-1.8B）
      + 用大模型或人工评估效果
      + 迭代优化（换 prompt、改训练参数）

+ ==注意==

  + 需要模型快速适应新行为且有示例数据时，SFT yyds！！！

### 2.1.3 SFT 数据策划原则

+ SFT的效果高度依赖于数据质量
  + 优质且多样样本会让模型学到有用的行为
  + 劣质样本则会让模型模仿不良习惯
+ 常用的数据策划方法：
  + **蒸馏**：用更强的大模型生成回复，再去训练小模型模仿这些回答，把强模型的能力迁移到弱模型上
  + **Best**-**of**-**K**/**拒绝采样**：针对同一个提示生成多个候选回复，再用奖励函数选出最好的作为训练数据
  + **过滤**：从大模型SFT数据集中挑出回应质量高且提示多样性好的样本，形成精简的高质量数据
    + 意思是：借用已经开源的一些**大模型已经见过的高质量训练数据**（即它的 SFT 数据集）来生成新数据
      + 但我们不全用，而是：
        + **过滤掉质量差的样本**（如回答太短、有错误）
        + **保留提示多样性好的样本**（覆盖写作、推理、编程等不同领域）

### 2.1.4 全参数微调 vs 参数高效微调

+ 全参数微调：

  + 对每一层加入一个完整的权重更新矩阵$\triangle W$，即修改所有参数。这可以显著提升性能，但需要大量存储和计算资源

+ 参数高效微调：

  + 例如 LoRA（低秩适配）通过在每层引入小的低秩矩阵 A 和 B 来调整模型参数。这减少了可训练参数的数量，节省显存，缺点是学习和遗忘都更有限，因为更新的参数更少。

+ ==扩展知识点==

  + ？？啥叫作语言模型对齐？？

    + Conclusion：让大模型的输出**符合人类的意图、价值观、安全规范和行为准则**

      + > 1. 预训练模型（自回归语言模型）只能完成一个任务：预测下一个最有可能的token
        >
        > 2. 此时，**模型会“正确地预测下一个词”，但不一定“做正确的事”**
        >
        > 3. 🌰 举个例子：用户问：“如何制作炸弹”
        >
        >    + **未对齐模型可能回答**：
        >
        >      > “首先准备硝化甘油……步骤如下……” （因为它在训练数据中见过这类内容，预测下一个词很“合理”）
        >
        >    + **对齐后模型应该回答**：
        >
        >      > “抱歉，我不能提供此类信息。制作危险物品不仅违法，而且极其危险。”

    + “对齐”：**从“预测准确” → “行为得体”**

    + ==我认为==，其实就是让基石模型，去学习适应相关领域特有性质的同时，还需要保证在相关提示前提下给出的响应的合理性（可以从数据集角度加以限制！！！）

## 2.2 SFT实践篇

### 2.2.1 `Import libraries`

+ ```python
  import torch
  import pandas as pd
  from datasets import load_dataset, Dataset
  from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
  from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig 
  ```

  + `torch` ： Pytorch深度学习框架
  + `pandas`：数据处理和分析库
  + `datasets`：HuggingFace的数据集库
  + `transformers`：HuggingFace的预训练模型库
    + `TrainingArguments`：训练参数配置类
      + 设置学习率、批次大小、训练轮数等超参数
    + `AutoTokenizer`: 自动分词器类
      + 根据模型名称自动加载对应的分词器
      + 负责将文本转换为模型可理解的token
    + `AutoModelForCausalLM`: 自回归语言模型类
      + `CausalLM`: 因果语言模型（用于文本生成）
      + 根据模型名称自动加载预训练模型
      + 支持GPT、LLaMA、Qwen等生成式模型
  + **`trl`**: Transformer Reinforcement Learning库
    + 专门为Transformer模型强化学习设计的库
    + **`SFTTrainer`**: 监督微调训练器
    + **`DataCollatorForCompletionOnlyLM`**: 数据整理器
      + `CompletionOnlyLM`: 只计算生成部分损失的整理器
      + 功能：在训练时只计算assistant回复部分的损失，忽略用户输入的损失
    + **`SFTConfig`**: 监督微调配置类
      + 继承自TrainingArguments，专门为SFT设计的配置
      + 包含SFT特有的超参数设置

### 2.2.2 `Setting up helper functions`

+ ```python
  def generate_responses(model, tokenizer, user_message, system_message=None, 
                         max_new_tokens=100):
      # Format chat using tokenizer's chat template
      messages = [] #创建一个空列表 messages，用于存储对话的历史消息
      #如果提供了 system_message（例如：“你是一个有用的助手。”），就将其以 system 角色添加到消息历史中。
      if system_message:
          messages.append({"role": "system", "content": system_message})
      
      # We assume the data are all single-turn conversation
      #将用户的输入消息作为 user 角色添加到对话中。
      messages.append({"role": "user", "content": user_message})
          
      #使用分词器内置的 chat template 来格式化整个对话。
      #prompt 是一个字符串，包含了完整的 system + user + assistant header 的模板文本。
      prompt = tokenizer.apply_chat_template(
          messages,	
          tokenize=False,  #不直接返回 token IDs，而是返回原始字符串格式的 prompt。
          add_generation_prompt=True,  #在最后加上生成起始标记，告诉模型“现在轮到你回复了”。
          enable_thinking=False,  #某些模型支持“思维链”模式（如 DeepSeek-R1），若设为 True 可能会触发 thinking 模式。此处禁用。
      )
  	
      # 将格式化后的 prompt 字符串通过分词器转换为模型可接受的输入张量。
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      # return_tensors="pt"：返回 PyTorch 张量（Tensor）。
     	# .to(model.device)：确保输入数据与模型处于同一设备（CPU/GPU），避免运行时报错。
      
      # Recommended to use vllm, sglang or TensorRT
      # 在无梯度模式下进行推理生成。 
      with torch.no_grad():
          outputs = model.generate(
              **inputs,    #传入编码后的输入（input_ids, attention_mask 等）
              max_new_tokens=max_new_tokens,  #最多生成 100 个新 token。
               do_sample=False, #使用贪婪解码（greedy decoding），不采样，保证输出确定性。
            # .to(model.device)：确保输入数据与模型处于同一设备（CPU/GPU），避免运行时报错。
              pad_token_id=tokenizer.eos_token_id, #设置填充符 ID（当 batch size > 1 时有用），这里用 EOS 作为 pad
         
              eos_token_id=tokenizer.eos_token_id, #遇到 EOS（结束符）时停止生成
          )
      input_len = inputs["input_ids"].shape[1] #获取输入部分的 token 数量（即 prompt 的长度） 
      generated_ids = outputs[0][input_len:] #从输出中切片，只保留新生成的部分（去掉输入 prompt 对应的 tokens）。
      # 从 input_len 开始截取，即模型真正生成的内容
      
      # 将生成的 token ID 解码为人类可读的文本。
      response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
      # skip_special_tokens=True：跳过特殊 token（如 <|endoftext|>, <|im_end|> 等）。
  	
      return response
  ```

  + 作用：使用一个预训练的语言模型（如 Llama、Qwen 等）和对应的分词器（Tokenizer），根据用户输入的消息（`user_message`）和可选的系统提示（`system_message`），生成模型的回复。
  + 适用于**单轮对话**（single-turn conversation）场景。

+ ```python
  # 它接收一组问题（questions），依次让模型回答，并将每个问题和对应的回答格式化输出到控制台，便于人工查看或评估模型行为。
  def test_model_with_questions(model, tokenizer, questions, 
                                system_message=None, title="Model Output"):
      print(f"\n=== {title} ===")
      
      # 遍历 questions 列表中的每一个问题。
      # enumerate(questions, 1)：为每个问题加上从 1 开始的序号（i），避免从 0 开始编号。
      # 例如：questions = ["你好吗？", "地球是什么？"]
  	# i=1, question="你好吗？"
  	# i=2, question="地球是什么？"
      for i, question in enumerate(questions, 1):
          response = generate_responses(model, tokenizer, question, 
                                        system_message)
          print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
  ```

+ ```python
  # 加载一个预训练的语言模型（如 Llama、Qwen 等）及其对应的分词器（Tokenizer），并进行基本配置（如设备放置、填充 token 设置、聊天模板定义），以便后续用于推理或对话生成。
  def load_model_and_tokenizer(model_name, use_gpu = False):
      
      # Load base model and tokenizer
      # 使用 Hugging Face 的 AutoTokenizer 自动根据模型名称加载对应的分词器。
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      # 加载指定名称的用于文本生成的自回归模型
      model = AutoModelForCausalLM.from_pretrained(model_name)
      
      if use_gpu:
          model.to("cuda")
      
      # 检查当前分词器是否已经内置了 chat template（对话模板）。
      # 如果没有（比如老模型或某些未更新的 tokenizer），则手动设置一个默认模板。
      if not tokenizer.chat_template:
          # 为分词器设置一个 Jinja2 格式的 自定义对话模板。
          tokenizer.chat_template = """{% for message in messages %}
                  {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                  {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                  {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                  {% endif %}
                  {% endfor %}"""
      
      # Tokenizer config
      # 配置分词器的 padding token（填充符）。
      if not tokenizer.pad_token:
          tokenizer.pad_token = tokenizer.eos_token
          
      return model, tokenizer
  ```

+ ```python
  # 从一个对话格式的数据集中提取前3个样本，并以表格形式可视化显示用户的提问和助手的回复。
  def display_dataset(dataset):
      # Visualize the dataset 
      # 初始化一个空列表 rows，用来存储将要展示的每一行数据（每个样本一条记录）
      rows = [] # 每个元素会是一个字典，包含 'User Prompt' 和 'Assistant Response' 两个键。
      for i in range(3):
          example = dataset[i] #获取数据集中第 i 个样本。每个样本是一个字典。
          # 从当前样本的 messages 列表中，找到第一个角色为 'user' 的消息，并提取其 content。
          user_msg = next(m['content'] for m in example['messages']
                          if m['role'] == 'user')
   		# next(...) 只返回第一个匹配项 —— 适合单轮对话或只关心第一轮用户输入的情况。
          assistant_msg = next(m['content'] for m in example['messages']
                               if m['role'] == 'assistant')
          rows.append({
              'User Prompt': user_msg,
              'Assistant Response': assistant_msg
          })
      
      # Display as table
      # 将 rows 列表转换为一个 Pandas 的 DataFrame 表格对象。
      df = pd.DataFrame(rows)
      pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
      display(df)
  ```

### 2.2.3 `Load base model & test on simple questions`

+ ```python
  USE_GPU = False
  
  questions = [
      "Give me an 1-sentence introduction of LLM.",
      "Calculate 1+1-1",
      "What's the difference between thread and process?"
  ]
  ```

+ ```python
  model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen3-0.6B-Base", USE_GPU)
  
  test_model_with_questions(model, tokenizer, questions, 
                            title="Base Model (Before SFT) Output")
  
  del model, tokenizer
  ```

### 2.2.4 `SFT results on Qwen3-0.6B model`

+ ```python
  model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen3-0.6B-SFT", USE_GPU)
  
  test_model_with_questions(model, tokenizer, questions, 
                            title="Base Model (After SFT) Output")
  
  del model, tokenizer
  ```

### 2.2.5 `Doing SFT on a small model`

+ ```python
  model_name = "./models/HuggingFaceTB/SmolLM2-135M"
  model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)
  ```

+ ```python
  train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
  if not USE_GPU:
      train_dataset=train_dataset.select(range(100))
  
  display_dataset(train_dataset)
  ```

+ ```python
  # SFTTrainer config 
  sft_config = SFTConfig(
      learning_rate=8e-5, # Learning rate for training. 学习率
      num_train_epochs=1, #  Set the number of epochs to train the model.训练轮数
      per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 每个 GPU（或 CPU）在一次前向传播中处理的样本数量。
      gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.梯度累积步数
      gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.梯度检查点
      logging_steps=2,  # Frequency of logging training progress (log every 2 steps).日志记录频率
  
  )
  ```

  + ==注意==SFT 一般不建议多轮训练（>3），容易导致模型“死记硬背”而丧失泛化能力；
  + 每设备训练批大小：批次越小，梯度越不稳定（噪声大），更新更频繁；
  + 梯度累积步数：在执行一次参数更新前，先累积多个批次的梯度
    + 模拟更大的 batch size（等效 batch size = `per_device_train_batch_size × gradient_accumulation_steps = 1×8=8`）
    + 在显存有限时实现大批次训练效果（更稳定梯度）
    + 这是**显存不足时的标准技巧**！！！！
  + 梯度检查点：通过牺牲计算时间来节省显存。
    + 不保存中间激活值，而是重新计算；

+ ```python
  sft_trainer = SFTTrainer(
      model=model,
      args=sft_config,
      train_dataset=train_dataset, 
      processing_class=tokenizer,
  )
  sft_trainer.train()
  ```

### 2.2.6 `Testing training results on small model and small dataset`

+ ```python
  if not USE_GPU: # move model to CPU when GPU isn’t requested
      sft_trainer.model.to("cpu")
  test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                            title="Base Model (After SFT) Output")
  ```

# 第三章 直接偏好优化理论及实践

## 3.1 直接偏好优化基础理论（Basics of DPO）

> 背景：
>
> ​	在DPO出现之前，对齐语言模型与人类偏好主要依赖**强化学习从人类反馈**（Reinforcement Learning from Human Feedback, RLHF）
>
> + **RLHF的问题**：
>   + 强化学习过程复杂、不稳定，超参数敏感。
>   + 需要额外的奖励模型和参考模型，增加了训练和部署的复杂性。
>   + PPO等算法在语言模型上容易出现训练崩溃或模式重复。

### 3.1.1 概述

+ **直接偏好优化（Direct Preference Optimization, DPO）**

  + 从正面和负面响应中进行对比学习的方法

  + > DPO 将“人类偏好”直接转化为损失函数，无需训练奖励模型和强化学习。==即==**在一定条件下，强化学习中的策略优化目标可以转化为一个简单的分类损失函数**

+ 针对用户可能会问的问题，至少需要准备两个回复，以便使得直接偏好优化(DPO)起作用

  + 希望其去模仿偏好的样本

+ Target：

  + 最小化对比损失
  + 对负面回复进行惩罚，并且鼓励正面回复

### 3.1.2 DPO损失

+ <font color=red>DPO损失实际上是对**重新参数化奖励模型的奖励差异**的交叉熵损失</font>

​									$\mathcal{L}\_{\text{DPO}} = -\log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_{\text{pos}} \mid x)}{\pi_{\text{ref}}(y_{\text{pos}} \mid x)} - \log \frac{\pi_\theta(y_{\text{neg}} \mid x)}{\pi_{\text{ref}}(y_{\text{neg}} \mid x)} \right) \right)$

+ 整体：对数差值---的---sigmoid函数---的---负对数【两个前后的对数差值，分别关注正样本和负样本】

  +  $$\sigma$$ 实际上就是sigmoid函数

  +  $$\beta$$ 是一个非常重要的超参数

     +  $$\beta$$ 值越高，这个对数差值就越重要。

  +  有两个概率比值的对数。
     + 前一块为正样本相关的分数，后一块为负样本相关的分数
     + 分子即 $$\pi_\theta$$ ，是微调后的模型
       + 含义：对于微调后的模型，在给定提示的情况下，产生正面回复的概率是多少。
     + 分母即$\pi_{ref}$是一个参考模型，**它是原始模型的副本，权重固定，不可调整**。
       + 含义：对于原始模型，在给定提示的情况下，产生那些正面回复的概率。同样，对于负样本，我们也有对数比值，其中 $$\pi_\theta$$ 是你微调后的模型， $$ \theta$$ 是你在这里想要调整的参数。而 $$\pi$$ 是一个固定的参考模型，可以是原始模型的副本。    

+ 本质上，这个对数比值项可以被看作是奖励模型的重新参数化。
  + 如果你将其<font color=red>视为</font>奖励模型，那么这个DPO损失实际上就是正样本和负样本之间奖励差异的sigmoid函数
+ 本质上，DPO试图最大化正样本的奖励，并最小化负样本的奖励。

### 3.1.3 DPO的最佳使用场景

+ **改变模型行为**:
  + 小的修改：
    + 改变模型特性
    + 使模型在多语言响应、指令遵循能力方面表现更好
    + 改变模型一些与安全相关的响应
+ **提升模型能力**
+ ==解析==
  + 模型行为？？工程化？？
    + **行为**是指在实际交互中具体表现出来的**动作或输出**，这<font color=red>取决于</font>模型的能力
    + 例如：
      + **遵循指令**：按照用户的明确要求进行回答或操作。
      + **拒绝回答**：对不安全、违法或超出范围的请求说“不”。
      + **创造内容**：写故事、诗歌、邮件等。
      + **表现出偏见或错误**：由于训练数据的偏差，可能生成带有偏见或不准确的信息。
      + **“幻觉”(Hallucination)**：生成看似合理但事实上错误或编造的内容。
  + 模型能力？？
    + **能力**指的是模型**理论上能够做什么**，也就是它的**潜在技能或功能范围**。
    + 通常由模型的**架构**、**训练数据**、**参数规模**和**训练方**法决定
    + 例如：
      + **语言理解**：理解文本的含义、情感、意图等。
      + **语言生成**：根据提示生成连贯、符合语法的文本。
      + **知识记忆**：在训练过程中学习到的大量事实性知识（截至其训练数据的时间点）。
      + **推理能力**：进行逻辑推理、数学计算、因果推断等。
      + **多语言能力**：理解和生成多种语言。
      + **代码生成**：编写和理解编程代码。
      + **上下文学习 (In-context Learning)**：通过少量示例或指令快速适应新任务。
  + ==我认为==
    + 能力是指“能做什么”，行为是指“实际做了什么”
    + 模型的能力是模型的行为的基石，模型的行为是模型的能力的延申

### 3.1.4 DPO的数据策划原则

+ **一种校正方法**：

  + **过程**：
    + 由原始模型生成回复
    + 将该回复作为一个主动样本
    + 对其改进之后，将改进之后的回复，作为一个**正向回复**
    + 这种基于纠正的方法，==自动==创建大规模、高质量的对比数据
  + <font color=red>疑惑？？？</font>：为啥是自动？？明明前文和例子中都是人为去进行改进的欸~！
    + 可以理解为，人为修改是一个`template`，人工只是参与**设计纠正策略或规则**，此后可以用“自动化”或者“半自动化”进行替代，而不是逐条人工修改
    + “**从人工示范到自动构造**”的范式升级
    + 高级的“自动化”方式：
      + 使用另一个强模型（如 GPT-4）作为“标注器”
      + 基于模板或规则生成正向回复
      + 使用强化学习或打分模型筛选与修正

+ **在线或策略内DPO的一种特殊情况**

  + **方案**：从模型自身的分布中生成**许多**正向和负向示例

  + 过程：

    + 针对同一个提示，让当前模型生成多个回复
    + 从众多回复中收集最佳回复作为正样本，最差回复作为负样本【⚠️要注意甄别！！！】
    + ==之后==你再判断哪个回复更好，哪个回复更差

  + <font color=red>疑惑？？？</font>：为什么还需要再去判断哪个回复好，哪个回复坏呢，不是已经收集最佳回复作为正样本了吗？

    + 不是“已经收集了最佳”，而是“收集了一堆回复，其中有一个是最佳”。
    + “最佳”和“最差”是**通过后续的判断过程从这批回复中筛选出来的**。

  + ⚠️⚠️：

    + > 避免过拟合
      >
      > 因为：1. 直接偏好优化本质上是在进行某种奖励学习，它很容易过度拟合到一些捷径上。

### 3.1.5 DPO数学原理详解

+ 问题设定

  + > 我们希望训练一个语言模型 $\pi_{\theta}(y|x)$，使其输出更符合人类偏好。我们拥有一个数据集$D=\{(x,y_w,y_l)\}$ 其中：
    >
    > + $x$：提示(prompt)
    > + $y_w$：正面响应(winning)
    > + $y_l$：负面响应(losing)

  + 目标🎯：**最大化模型生成$y_w$而非$y_l$的概率**，同时保持与原始模型不偏离（防止过拟合）

+ 传统方法：RLHF 中的 PPO 目标

  + 目标函数：$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[  \log \sigma \left( \beta \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right) - \lambda D_{\text{KL}}\left( \pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x) \right) \right]$
  + 需要：
    + 显式训练一个奖励模型 $r_\theta$
    + 维护一个参考策略 $\pi_{ref}$
    + 使用强化学习更新策略
  + ==引入DPO核心思想==：**绕过奖励模型和强化学习，直接在偏好数据上优化策略**。

+ Bradley-Terry 模型：建模人类偏好

  + DPO 假设人类偏好服从 **Bradley-Terry 模型**：

    + > $P(y_w \succ y_l \mid x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}}$

    + 它假设每个响应 $y$对应一个“隐含奖励” $r(x,y)$，奖励越高，越可能被选择。

    + ‼️**给定提示 $x$，人类更喜欢 $y_w$ 而不是$y_l$ 的概率**。

    + 为啥要取e的指数？？？

      + 将奖励映射到正实数空间，用于归一化比较💡

  + 奖励越大，被选中的概率越大！


+ 奖励函数与策略的关系（DPO 的🧠核心洞察）

  + > $r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + r_{\text{ref}}(x)$
    >
    > + $\beta$：温度参数（scalar），控制 KL 正则的强度，通常$\beta$>0
    > + $r_{ref}(x)$：仅依赖于输$x$的偏置项，不影响 $y_w$和 $y_l$ 的相对比较

  + 🌹核心突破🐮🍺：

    + **不需要显式训练一个奖励模型**，而是通过当前策略 $\pi_\theta$与参考策略$\pi_{ref}$ 的对数比值来**隐式定义奖励**。

  + 解析：

    + $\pi_\theta(y|x)>\pi_{ref}(y|x)$，说明当前模型比参考模型更愿意生成$y$，因此$r(x,y)$更高。
    + $\pi_\theta(y|x)<\pi_{ref}(y|x)$，说明当前模更不愿意生成$y$，因此$r(x,y)$更低。
    + $\beta$控制这个“偏离奖励”的放大程度：$\beta$越大，模型越激进；反之，越保守

+ 结合上面俩式子


  + 分别计算$r(x,y_w)$和$r(x,w_l)$


    + > 1. $r(x,y_w)=\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + r_{\text{ref}}(x)$
      > 2. $r(x,y_l)=\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + r_{\text{ref}}(x)$

  + 分别计算$e^{r(x,y_w)}$和$e^{r(x,w_l)}$

    + > 1. $e^{r(x,y_w)}=e^{\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + r_{\text{ref}}(x)}=e^{{\text{ref}}(x)} \cdot (\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)})^\beta$
      > 2. $e^{r(x,y_l)}=e^{\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + r_{\text{ref}}(x)}=e^{{\text{ref}}(x)} \cdot (\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})^\beta$

  + 代入**Bradley-Terry 模型**：


    + > 1. 已知模型公式为：$P(y_w \succ y_l \mid x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}}$
      > 2. $P(y_w \succ y_l \mid x) = \frac{1}{1 + e^{r(x,y_l)-r(x,y_w)}}=\frac{1}{1+e^{\beta(\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)})}} = \frac{1}{1+(\frac{\pi_\theta(y_l|x) \div \pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_w|x) \div \pi_{\text{ref}}(y_w|x)})^\beta}$

  + 使用`sigmoid`恒等式：


    + > 1. ‼️恒等式：$\frac{1}{1+e^{-z}} = \sigma(z)$，即$\frac{1}{1+a} = \sigma(-\log a)$
      > 2. 令：$a = (\frac{\pi_\theta(y_l|x) \div \pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_w|x) \div \pi_{\text{ref}}(y_w|x)})^\beta$
      > 3. 🍇：$P(y_w \succ y_l \mid x) = \frac{1}{1+a} = \sigma(-\log a) = \sigma(-\beta \log(\frac{\pi_\theta(y_l|x) \div \pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_w|x) \div \pi_{\text{ref}}(y_w|x)})) = \sigma(\beta \log(\frac{\pi_\theta(y_w|x) \div \pi_{\text{ref}}(y_w|x)}{\pi_\theta(y_l|x) \div \pi_{\text{ref}}(y_l|x)})) $
      > 4. ☀️：$P(y_w \succ y_l \mid x) = \sigma(\beta \log(\frac{\pi_\theta(y_w|x) \div \pi_{\text{ref}}(y_w|x)}{\pi_\theta(y_l|x) \div \pi_{\text{ref}}(y_l|x)})) = \sigma(\beta(\log \frac{\pi_\theta(y_w|x)}{ \pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{ \pi_{\text{ref}}(y_l|x)}))\\= \sigma(\beta(\log \frac{\pi_\theta(y_w|x)}{ \pi_\theta(y_l|x)} - \log \frac{\pi_{\text{ref}}(y_w|x)}{ \pi_{\text{ref}}(y_l|x)}))$
      >
      > > $\log \frac{\pi_\theta(y_w|x)}{ \pi_\theta(y_l|x)}$：表示**当前模型**认为$y_w$相对于$y_l$的“偏好强度”;
      > >
      > > $\log \frac{\pi_{ref}(y_w|x)}{ \pi_{ref}(y_l|x)}$：表示**参考模型**认为$y_w$相对于$y_l$的“偏好强度”;
      > >
      > > - **两者之差：当前模型比参考模型多“强化”了多少偏好**

  + $\sigma(z)$将实数映射到(0, 1)

+ DPO损失函数：


  + > $$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ 
\log \sigma\left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log \frac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) 
\right]
$$

  + ✅含义：


    + **最终训练目标**。我们希望最大化人类偏好发生的概率，等价于最小化其**负对数似然**。
    + $\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}$：表示对整个偏好数据集 DD 取期望（即平均损失）

  + 

### 3.1.6 ==思考==

+ DPO到底有没有奖励模型，要是没有的话，那“DPO损失实际上是对重新参数化奖励模型的奖励差异的交叉熵损失”这句话是什么意思呢？
  + **DPO 不显式地训练或使用一个独立的“奖励模型”（Reward Model），但它隐式地依赖于一个从参考模型和KL散度中推导出来的“隐式奖励模型”。**
  + 隐式奖励模型，由上述公式推导可知，是根据当前策略 $\pi_\theta$与参考策略$\pi_{ref}$ 的对数比值来**隐式定义奖励**⚠️

# 第四章 在线强化学习理论及实践

# 第五章 课程总结
