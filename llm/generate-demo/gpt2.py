
"""
这段代码的核心是将输入文本通过 GPT-2 进行推理，获取下一个可能的词，并展示了模型如何在自回归过程中生成文本。

执行结果参考：
df@moss:~/codes/ai/ai-labs/llm$ python3   gpt2-demo.py
Encoded tokens: {'input_ids': tensor([[ 64, 275, 269, 288]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
Decoded text: a b c d

---------------------------------------- hidden_states : tensor([[[-9.2553e-02,  2.6298e-01,  1.7400e-02,  ..., -3.3709e-02,
          -6.0129e-01, -1.1269e-01],
         [-3.6091e-01, -6.9428e-04,  6.4340e-01,  ..., -1.1169e-01,
          -2.6098e-01,  1.8140e-01],
         [-3.3533e-01,  3.6901e-01,  5.0720e-01,  ..., -7.3986e-02,
          -4.7744e-01,  7.4971e-01],
         [-6.6894e-02,  1.3590e-01,  8.2749e-01,  ...,  1.7294e-01,
          -1.3736e-01,  1.0548e+00]]], device='cuda:0') ,torch.Size([1, 4, 1024])
 打印位置：
 /usr/local/lib/python3.10/dist-packages/transformers/models/gpt2/modeling_gpt2.py , forward(), line 1094
----------------------------------------

----------------------------------------  lm_logits tensor([[[-68.0251, -65.0669, -69.1020,  ..., -77.5058, -74.3794, -67.8483],
         [-70.7141, -69.9348, -71.2718,  ..., -80.5144, -78.5028, -70.4597],
         [-10.7032,  -8.2438,  -8.5123,  ..., -18.6245, -16.8126,  -8.7227],
         [  0.8500,   4.0325,   1.7252,  ...,  -5.7310,  -4.9616,   3.0226]]],
       device='cuda:0'), torch.Size([1, 4, 50257])
 打印位置：
 /usr/local/lib/python3.10/dist-packages/transformers/models/gpt2/modeling_gpt2.py , forward(), line 1101
----------------------------------------

---------------------------------------- labels : None
 打印位置：
 /usr/local/lib/python3.10/dist-packages/transformers/models/gpt2/modeling_gpt2.py , forward(), line 1104
----------------------------------------
Logits shape: torch.Size([1, 4, 50257])
output.logits: tensor([[[-68.0251, -65.0669, -69.1020,  ..., -77.5058, -74.3794, -67.8483],
         [-70.7141, -69.9348, -71.2718,  ..., -80.5144, -78.5028, -70.4597],
         [-10.7032,  -8.2438,  -8.5123,  ..., -18.6245, -16.8126,  -8.7227],
         [  0.8500,   4.0325,   1.7252,  ...,  -5.7310,  -4.9616,   3.0226]]],
       device='cuda:0')
Next token logits shape: tensor([[ 0.8500,  4.0325,  1.7252,  ..., -5.7310, -4.9616,  3.0226]],
       device='cuda:0') torch.Size([1, 50257])
Predicted next token:  e

"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


path = '../model/gpt2-medium'
# 加载 tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_text = "a b c d"

# 编码输入文本
tokens = tokenizer(input_text, return_tensors="pt")
print("Encoded tokens:", tokens)  # 打印编码后的 token 和对应的输入 ID

input_ids = tokens.input_ids.to(device)
# 将输入的 token ID 解码为原始文本
decoded_text = tokenizer.decode(input_ids[0])
print("Decoded text:", decoded_text)

# 使用模型预测下一个词的 logits
with torch.no_grad():
    output = model(input_ids)
    # 打印输出 logits 的维度
    print("Logits shape:", output.logits.shape)  # [1, seq_len, vocab_size]
    print(f'output.logits: {output.logits}')
    # 获取最后一个位置的 logits
    next_token_logits = output.logits[:, -1, :]
    print("Next token logits shape:", next_token_logits, next_token_logits.shape)

# 取概率最高的 token 作为预测词
predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
predicted_token = tokenizer.decode(predicted_token_id)

print("Predicted next token:", predicted_token)


"""https://github.com/huggingface/transformers/blob/768f3c016eec88a00f2a991c7017a8a5423c4b06/src/transformers/models/gpt2/modeling_gpt2.py#L1286C6-L1286C7
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    ........
        hidden_states = transformer_outputs[0]  # 最后一层输出

      # Set device for model parallelism
      if self.model_parallel:
          torch.cuda.set_device(self.transformer.first_device)
          hidden_states = hidden_states.to(self.lm_head.weight.device)

      lm_logits = self.lm_head(hidden_states) # 转词典维度
"""