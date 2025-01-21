from vllm import LLM, SamplingParams
import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

def main(prompts, model_path):
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,      # 控制生成的随机性，越低越确定性
        top_k=-1,             # 限制候选词的数量，-1 表示不限制
        max_tokens=100,       # 最大生成长度
    )

    # 创建一个 LLM 对象，加载模型
    llm = LLM(model=model_path, trust_remote_code=True)

    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    # 输出生成结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")

if __name__ == '__main__':
    # 示例输入
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "明月别枝惊鹊，",
    ]
    
    # 指定模型路径
    model_path = 'model/Qwen/Qwen2-5-1-5B-Instruct'
    
    # 调用主函数
    main(prompts, model_path)
