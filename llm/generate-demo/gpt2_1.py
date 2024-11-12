"""
改进代码

接上，这段代码有问题么？能作为示例代码展示目前先进的大模型生成原理的demo 么？给出关键证据，如果认为不行，给一个示例代码???

    这段代码整体上是能够运行的，并且能够演示 GPT-2 模型如何基于输入文本生成下一个可能的词。然而，考虑到 **当前先进的大模型生成原理**（如 GPT-3、GPT-4 等更大模型的生成），这段代码的示例存在一些局限性。以下是一些原因，以及改进建议：

### 存在的问题：

1. **生成过程仅限于单个 token 的预测**：
   - 这段代码只预测了输入文本 `"a b c d"` 的 **下一个词**。而在实际的生成任务中，我们往往会需要进行多个 token 的生成，直到满足某个条件（如生成结束标记或达到最大长度）。
   - **缺少连贯性**：模型只预测了一个 token，并没有生成后续的词。现实中，大模型如 GPT-3 通常会基于先前生成的 tokens 进行 **自回归生成**，以确保生成的文本连贯。

2. **生成的质量较低**：
   - `argmax` 只是选择概率最大的 token，忽略了 **温度采样**、**Top-k 采样**、**Top-p (nucleus) 采样** 等策略，这些策略可以增加生成文本的多样性和质量。
   - 对于更大、更复杂的模型，如 GPT-3，生成过程更依赖于这些采样技巧。

3. **没有处理 `pad_token_id` 或结束标记**：
   - 对于大模型的生成任务，通常会使用 `pad_token_id` 和 `eos_token_id`（结束标记），以便在生成过程中标记句子结束，防止无限循环生成不合适的输出。
   - 这段代码没有处理这些细节，可能导致不必要的冗长输出。

### 关键证据：

1. **先进的生成策略**：当前的高级语言模型通常使用温度采样、Top-k 采样、Top-p (nucleus) 采样等策略，而不是简单的 `argmax`。这些策略使得生成的文本更富有创意并且更符合自然语言的多样性。

2. **连续生成**：在实际应用中，模型通常需要生成多个词，并且基于先前生成的 token 进行预测。这个过程是 **递归的**，每次生成一个 token 后都会将它加入到上下文中，作为下一步生成的输入。

### 改进的示例代码：

以下是改进后的示例代码，演示如何生成多步文本，并且应用温度采样（可调整采样的随机性）和 Top-k 采样（限制候选词的数量）：

### 关键改进点：

1. **生成多个 token**：
   - 这段代码使用 `model.generate()` 来生成 **多步**（`max_length` 控制生成长度）的文本，而不仅仅是预测下一个 token。

2. **采样策略**：
   - `temperature` 控制生成的 **随机性**，较低的温度会使生成更为确定，较高的温度则增加生成文本的多样性。
   - `top_k` 限制每次生成时候候选词的数量，可以避免生成过于随机或无意义的词。

3. **启用采样**：
   - `do_sample=True` 启用了采样，而不是简单地选择最大概率的词（`argmax`）。这会导致生成的文本更加自然，并有更多的变化。

4. **使用 `generate`**：
   - `model.generate()` 是 Hugging Face 提供的一个高级接口，专门用于文本生成任务，自动处理了很多底层的细节，比如序列的递归生成、标记结束符等。

### 总结：

尽管原始代码能够执行一个简单的单步预测，但它并没有展示当前先进大模型（如 GPT-3）的 **生成能力**。
上面的改进版本更符合现代大模型生成文本的实际应用，使用了温度采样和 Top-k 采样，支持递归生成，并且能够生成多步文本。这使得生成的文本更加自然、富有创意，并且符合实际应用中的需求。
    
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random

path = '../model/gpt2-medium'
# 加载 tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_text = "a b c d"
# 编码输入文本
tokens = tokenizer(input_text, return_tensors="pt")
input_ids = tokens.input_ids.to(device)

# 设置生成参数
max_length = 20  # 生成的最大长度
temperature = 0.7  # 控制输出的多样性，越高越随机
top_k = 50  # Top-k 采样

# 使用模型生成文本
with torch.no_grad():
    model.eval()  # 切换到评估模式
    output = model.generate(
        input_ids,  # 输入的 token ids
        max_length=max_length,  # 最大生成长度
        temperature=temperature,  # 温度采样
        top_k=top_k,  # Top-k 采样
        do_sample=True,  # 启用采样，使用随机性而非直接选择最大概率
        num_return_sequences=1  # 生成一个序列
    )

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)  #  Generated text: a b c d e f g h i j k l m n o p q r s t
