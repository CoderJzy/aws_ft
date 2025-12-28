from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 模型路径
model_id = os.environ.get("model_id", "Qwen/Qwen3-1.7B")

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# 加载数据
dataset = load_dataset("json", data_files={"train": "/opt/ml/input/data/train/train.jsonl"})["train"]

# tokenization，附加 labels
def tokenize(example):
    output = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    output["labels"] = output["input_ids"].copy()
    return output

tokenized_dataset = dataset.map(tokenize, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2
)

# 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
trainer.train()
