from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

model_name = "/root/align-anything/outputs/qwen_2_5_dpo_trsize5k_5ep/slice_end"
device ='npu:0'

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("parquet", data_files={"validation": "/data/align_anything_t2t/val_1k.parquet"})
validation_data = dataset["validation"].shuffle(seed=42)
# subset = validation_data.select(range(200))
subset = validation_data

results = []

for item in tqdm(subset, desc="Generating", total=len(subset)):
    instruction = item["question"]
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    results.append({
        "instruction": instruction,
        "output": response
    })

# 保存结果
with open("dpo_1k_5k5ep_generation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
