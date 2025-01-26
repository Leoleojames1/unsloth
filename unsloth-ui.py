#!/usr/bin/env python3

"""
🦥 Unsloth Fine-Tuning UI

An interactive interface for fine-tuning models using Unsloth with Gradio.

Features:
- Model loading and configuration with common parameters
- Dataset preparation with multiple formats (alpaca, chatml, jsonl)
- LoRA fine-tuning with configurable hyperparameters
- Model saving in various formats (merged 16/4bit, LoRA adapter)
- GGUF quantization (16/8/4bit, custom)
- Hugging Face Hub integration
- Inference testing

Dataset formats supported:
- Alpaca (instruction/input/output columns)
- ChatML/ShareGPT (conversations)
- JSONL (messages)

Models supported:
- Llama 3/3.1/3.2
- Mistral/Mixtral
- Phi-3/4
- Gemma 2B/7B
- Qwen2
- Yi

Run with:
python unsloth-ui.py

    Written by:
        @sebdg, and modified by @lborcherding
Happy fine-tuning! 🦥
"""
import gradio as gr
from huggingface_hub import HfApi
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_sharegpt

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import time
import psutil
import platform
import os

hf_user = None
try:
    hfApi = HfApi()
    hf_user = hfApi.whoami()["name"]
except Exception as e:
    hf_user = "not logged in"

def get_human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


# get cpu stats
disk_stats = psutil.disk_usage('.')
print(get_human_readable_size(disk_stats.total))
cpu_info = platform.processor()
print(cpu_info)
os_info = platform.platform()
print(os_info)

memory = psutil.virtual_memory()

# Dropdown options
model_options = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit-instruct",
    "unsloth/gemma-2-27b-bnb-4bit",          # Gemma 2x faster!
    "unsloth/gemma-2-27b-bnb-4bit-instruct",          # Gemma 2x faster!
    "unsloth/Qwen2-1.5B-bnb-4bit",         
    "unsloth/Qwen2-1.5B-bnb-4bit-instruct",         
    "unsloth/Qwen2-7B-bnb-4bit",          
    "unsloth/Qwen2-7B-bnb-4bit-instruct",          
    "unsloth/Qwen2-72B-bnb-4bit",         
    "unsloth/Qwen2-72B-bnb-4bit-instruct",         
    "unsloth/yi-6b-bnb-4bit",          
    "unsloth/yi-34b-bnb-4bit",        
]
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

running_on_hf = False
if os.getenv("SYSTEM", None) == "spaces":
    running_on_hf = True

system_info = f"""\
- **System:** {os_info}
- **CPU:** {cpu_info} **Memory:** {get_human_readable_size(memory.free)} free of {get_human_readable_size(memory.total)}
- **GPU:** {gpu_stats.name} ({max_memory} GB)
- **Disk:** {get_human_readable_size(disk_stats.free)} free of {get_human_readable_size(disk_stats.total)}
- **Hugging Face:** {running_on_hf}
"""

model=None
tokenizer = None
dataset = None
max_seq_length = 2048

class PrinterCallback(TrainerCallback):
    step = 0
    def __init__(self, progress):
        self.progress = progress
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            #print(logs)
            pass
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.step = state.global_step
            self.progress(self.step/60, desc=f"Training {self.step}/60")
            #print("**Step ", state.global_step)

def formatting_prompts_func(examples, template_style):
    if template_style == "chatml":
        # Handle ShareGPT/ChatML format
        messages = examples["conversations"]
        texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) 
                for message in messages]
        return {"text": texts}
    
    elif template_style == "alpaca":
        # Handle Alpaca format
        instructions = examples["instruction"] 
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}{tokenizer.eos_token}

### Input:
{input}{tokenizer.eos_token}

### Response:
{output}{tokenizer.eos_token}"""
            texts.append(text)
        return {"text": texts}
    
    elif template_style == "jsonl":
        # Handle generic JSONL chat format
        messages = examples["messages"]
        texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) 
                for message in messages]
        return {"text": texts}

def load_model(initial_model_name, load_in_4bit, max_sequence_length, hub_token):
    global model, tokenizer
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=initial_model_name,
        max_seq_length=max_sequence_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        token=hub_token
    )
    
    template = get_model_template(initial_model_name)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=template
    )
    
    return (f"Model loaded with {template} template", 
            gr.update(visible=True, interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False))
    
def get_model_template(model_name):
    name = model_name.lower()
    templates = {
        "llama-3": "llama-3",
        "llama-31": "llama-3.1", 
        "llama-3.1": "llama-3.1",
        "mistral": "mistral",
        "gemma": "gemma",
        "phi-3": "phi-3",
        "phi-35": "phi-3",
        "phi-3.5": "phi-3",
        "phi-4": "phi-4",
        "qwen-2.5": "qwen-2.5",
        "qwen-25": "qwen-2.5", 
        "chatml": "chatml",
        "vicuna": "vicuna",
        "alpaca": "alpaca",
        "zephyr": "zephyr",
        "unsloth": "unsloth"
    }
    
    for key, template in templates.items():
        if key in name:
            return template
    return "llama-3.1"  # Default

def load_data(dataset_name, template_style, progress=gr.Progress()):
    global dataset, tokenizer
    
    progress(0.1, desc="Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    
    if template_style == "chatml":
        dataset = standardize_sharegpt(dataset)
    
    progress(0.5, desc="Formatting dataset...")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, template_style), 
        batched=True
    )
    
    # Create preview data
    preview_data = []
    for i in range(min(5, len(dataset))):
        if template_style == "alpaca":
            preview_data.append([
                f"Instruction: {dataset[i]['instruction']}\nInput: {dataset[i]['input']}", 
                dataset[i]['output']
            ])
        elif template_style in ["chatml", "jsonl"]:
            messages = dataset[i]['conversations']
            preview_data.append([
                messages[0]['value'],
                messages[1]['value'] if len(messages) > 1 else ""
            ])
    
    progress(1.0, desc="Dataset loaded")
    return (
        f"Data loaded: {len(dataset)} records", 
        gr.update(visible=True, interactive=True),
        gr.update(value=preview_data)
    )

def inference(input_text):
    FastLanguageModel.for_inference(model)
    
    # Parse input into instruction/input
    parts = input_text.split("# input")
    instruction = parts[0].replace("# instruction", "").strip()
    input_text = parts[1].strip() if len(parts) > 1 else ""
    
    messages = [
        {"role": "user", "content": f"{instruction}\n{input_text}"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        use_cache=True
    )
    
    result = tokenizer.batch_decode(outputs)[0]
    return result, gr.update(visible=True, interactive=True)

def save_model(model_name, hub_model_name, hub_token, gguf_16bit, gguf_8bit, gguf_4bit, gguf_custom, gguf_custom_value, merge_16bit, merge_4bit, just_lora, push_to_hub, progress=gr.Progress()):
    global model, tokenizer

    print("Starting save_model function")
    print(f"Model name: {model_name}")
    print(f"Hub model name: {hub_model_name}")
    print(f"GGUF 16bit: {gguf_16bit}, GGUF 8bit: {gguf_8bit}, GGUF 4bit: {gguf_4bit}")
    print(f"Merge 16bit: {merge_16bit}, Merge 4bit: {merge_4bit}, Just LoRA: {just_lora}")
    print(f"Push to hub: {push_to_hub}")
    
    quants = []
    current_quant = 0

    if gguf_custom:
        gguf_custom_value = gguf_custom_value
        quants.append(gguf_custom_value)
        print(f"Custom GGUF value: {gguf_custom_value}")
    else:
        gguf_custom_value = None
    
    if gguf_16bit:
        quants.append("f16")
    if gguf_8bit:
        quants.append("q8_0")
    if gguf_4bit:
        quants.append("q4_k_m")
    
    if merge_16bit:
        print("Merging model to 16bit")
        progress(current_quant/len(quants), desc=f"Pushing model merged 16bit {model_name} to HuggingFace Hub")
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_16bit",
        )
        if push_to_hub:
            print("Pushing merged 16bit model to HuggingFace Hub")
            model.push_to_hub_merged(hub_model_name, tokenizer, save_method="merged_16bit", token=hub_token)

    elif merge_4bit:
        print("Merging model to 4bit")
        progress(current_quant/len(quants), desc=f"Pushing model merged 4bit {model_name} to HuggingFace Hub")
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_4bit_forced",
        )
        if push_to_hub:
            print("Pushing merged 4bit model to HuggingFace Hub")
            model.push_to_hub_merged(hub_model_name, tokenizer, save_method="merged_4bit_forced",  token=hub_token)

    elif just_lora:
        print("Saving just LoRA")
        progress(current_quant/len(quants), desc=f"Pushing model merged lora {model_name} to HuggingFace Hub")
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="lora",
        )
        if push_to_hub:
            print("Pushing LoRA model to HuggingFace Hub")
            model.push_to_hub_merged(hub_model_name, tokenizer, save_method="lora",  token=hub_token)

    if push_to_hub:
        current_quant = 0
        for q in quants:
            print(f"Pushing model with quantization {q} to HuggingFace Hub")
            progress(current_quant/len(quants), desc=f"Pushing model {model_name} with {q} to HuggingFace Hub")
            model.push_to_hub_gguf(hub_model_name, tokenizer, quantization_method=q, token=hub_token)
            current_quant += 1
    print("Model saved successfully")
    return "Model saved", gr.update(visible=True, interactive=True)

def username(profile: gr.OAuthProfile | None):
    hf_user = profile["name"] if profile else "not logged in"
    return hf_user

# Create the Gradio interface
with gr.Blocks(title="Unsloth fine-tuning") as demo:
    if (running_on_hf):
        gr.LoginButton()
    # logged_user = gr.Markdown(f"**User:** {hf_user}")
    #demo.load(username, inputs=None, outputs=logged_user)
    with gr.Row():
        with gr.Column(scale=0.5):
            gr.Image("unsloth.png", width="300px", interactive=False, show_download_button=False, show_label=False, show_share_button=False)
        with gr.Column(min_width="550px", scale=1):
            gr.Markdown(system_info) 
        with gr.Column(min_width="250px", scale=0.3):
            gr.Markdown(f"**Links:**\n\n* [Unsloth Hub](https://huggingface.co/unsloth)\n\n* [Unsloth Docs](http://docs.unsloth.com/)\n\n* [Unsloth GitHub](https://github.com/unslothai/unsloth)")
    with gr.Tab("Base Model Parameters"):

        with gr.Row():
            initial_model_name = gr.Dropdown(choices=model_options, label="Select Base Model", allow_custom_value=True)
            load_in_4bit = gr.Checkbox(label="Load 4bit model", value=True)

        gr.Markdown("### Target Model Parameters")
        with gr.Row():
            max_sequence_length = gr.Slider(minimum=128, value=512, step=64, maximum=128*1024, interactive=True, label="Max Sequence Length")
        load_btn = gr.Button("Load")
        output = gr.Textbox(label="Model Load Status", value="Model not loaded", interactive=False)
        gr.Markdown("---")

    with gr.Tab("Data Preparation"):
        with gr.Row():
            dataset_name = gr.Textbox(label="Dataset Name", value="yahma/alpaca-cleaned")
            data_template_style = gr.Dropdown(
                label="Template", 
                choices=["alpaca", "chatml", "jsonl"],
                value="alpaca"
            )
        with gr.Row():
            gr.Markdown("Dataset Preview (first few records will be shown here after loading)")
            dataset_preview = gr.DataFrame(
                headers=["Input", "Output"],
                label="Dataset Preview"
            )
        gr.Markdown("---")
        output_load_data = gr.Textbox(label="Data Load Status", value="Data not loaded", interactive=False)
        load_data_btn = gr.Button("Load Dataset", interactive=True)
        load_data_btn.click(
            load_data, 
            inputs=[dataset_name, data_template_style], 
            outputs=[output_load_data, load_data_btn, dataset_preview]
        )

    with gr.Tab("Fine-Tuning"):
        gr.Markdown("""### Fine-Tuned Model Parameters""")
        with gr.Row():
            model_name = gr.Textbox(label="Model Name", value=initial_model_name.value, interactive=True)

        gr.Markdown("""### Lora Parameters""")

        with gr.Row():
            lora_r = gr.Number(label="R", value=16, interactive=True)
            lora_alpha = gr.Number(label="Lora Alpha", value=16, interactive=True)
            lora_dropout = gr.Number(label="Lora Dropout", value=0.1, interactive=True)

        gr.Markdown("---")
        gr.Markdown("""### Training Parameters""")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    per_device_train_batch_size = gr.Number(label="Per Device Train Batch Size", value=2, interactive=True)
                    warmup_steps = gr.Number(label="Warmup Steps", value=5, interactive=True)
                    max_steps = gr.Number(label="Max Steps", value=60, interactive=True)
                    gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=4, interactive=True)
                with gr.Row():
                    logging_steps = gr.Number(label="Logging Steps", value=1, interactive=True)
                    log_to_tensorboard = gr.Checkbox(label="Log to Tensorboard", value=True, interactive=True)

                with gr.Row():
                #     optim = gr.Dropdown(choices=["adamw_8bit", "adamw", "sgd"], label="Optimizer", value="adamw_8bit")
                    learning_rate = gr.Number(label="Learning Rate", value=2e-4, interactive=True)

                # with gr.Row():
                    weight_decay = gr.Number(label="Weight Decay", value=0.01, interactive=True)
                    # lr_scheduler_type = gr.Dropdown(choices=["linear", "cosine", "constant"], label="LR Scheduler Type", value="linear")
        gr.Markdown("---")

        with gr.Row():
            seed = gr.Number(label="Seed", value=3407, interactive=True)
            output_dir = gr.Textbox(label="Output Directory", value="outputs", interactive=True)
        gr.Markdown("---")

        train_output = gr.Textbox(label="Training Status", value="Model not trained", interactive=False)
        train_btn = gr.Button("Train", visible=True)

        def train_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, per_device_train_batch_size: int, warmup_steps: int, max_steps: int,
                gradient_accumulation_steps: int, logging_steps: int, log_to_tensorboard: bool, learning_rate, weight_decay, seed: int, output_dir, progress= gr.Progress()):
            global model, tokenizer
            print(f"$$$ Training model {model_name} with {lora_r} R, {lora_alpha} alpha, {lora_dropout} dropout, {per_device_train_batch_size} per device train batch size, {warmup_steps} warmup steps, {max_steps} max steps, {gradient_accumulation_steps} gradient accumulation steps, {logging_steps} logging steps, {log_to_tensorboard} log to tensorboard, {learning_rate} learning rate, {weight_decay} weight decay, {seed} seed, {output_dir} output dir")
            iseed = seed
            model = FastLanguageModel.get_peft_model(
                model,
                r = lora_r,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = "none",
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state=iseed,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
            progress(0.0, desc="Loading Trainer")
            time.sleep(1)
            trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc = 2,
                packing = False, # Can make training 5x faster for short sequences.
                callbacks = [PrinterCallback(progress)],
                args = TrainingArguments(
                    per_device_train_batch_size = per_device_train_batch_size,
                    gradient_accumulation_steps = gradient_accumulation_steps,
                    warmup_steps = warmup_steps,
                    max_steps = 60, # Set num_train_epochs = 1 for full training runs
                    learning_rate = learning_rate,
                    fp16 = not is_bfloat16_supported(),
                    bf16 = is_bfloat16_supported(),
                    logging_steps = logging_steps,
                    optim = "adamw_8bit",
                    weight_decay = weight_decay,
                    lr_scheduler_type = "linear",
                    seed = iseed,
                    report_to="tensorboard" if log_to_tensorboard else None,
                    output_dir = output_dir
                ),
            )
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
            trainer.train()
            progress(1, desc="Training completed")
            time.sleep(1)
            return "Model trained 100%",gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=True), gr.update(interactive=True)


        train_btn.click(train_model, inputs=[model_name, lora_r, lora_alpha, lora_dropout, per_device_train_batch_size, warmup_steps, max_steps, gradient_accumulation_steps, logging_steps, log_to_tensorboard, learning_rate, weight_decay, seed, output_dir], outputs=[train_output, train_btn])

    with gr.Tab("Save & Push Options"):

        with gr.Row():
            gr.Markdown("### Merging Options")
            with gr.Column():
                merge_16bit = gr.Checkbox(label="Merge to 16bit", value=False, interactive=True)
                merge_4bit = gr.Checkbox(label="Merge to 4bit", value=False, interactive=True)
                just_lora = gr.Checkbox(label="Just LoRA Adapter", value=False, interactive=True)
        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### GGUF Options")
            with gr.Column():
                gguf_16bit = gr.Checkbox(label="Quantize to f16", value=False, interactive=True)
                gguf_8bit = gr.Checkbox(label="Quantize to 8bit (Q8_0)", value=False, interactive=True)
                gguf_4bit = gr.Checkbox(label="Quantize to 4bit (q4_k_m)", value=False, interactive=True)
            with gr.Column():
                gguf_custom = gr.Checkbox(label="Custom", value=False, interactive=True)
                gguf_custom_value = gr.Textbox(label="", value="Q5_K", interactive=True)
        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Hugging Face Hub Options")
            push_to_hub = gr.Checkbox(label="Push to Hub", value=False, interactive=True)
            with gr.Column():
                hub_model_name = gr.Textbox(label="Hub Model Name", value=f"username/model_name", interactive=True)
                hub_token = gr.Textbox(label="Hub Token", interactive=True, type="password")
        gr.Markdown("---")
            
        # with gr.Row():
        #     gr.Markdown("### Ollama options")
        #     with gr.Column():
        #         ollama_create_local = gr.Checkbox(label="Create in Ollama (local)", value=False, interactive=True)
        #         ollama_push_to_hub = gr.Checkbox(label="Push to Ollama", value=False, interactive=True)
        #     with gr.Column():
        #         ollama_model_name = gr.Textbox(label="Ollama Model Name", value="user/model_name")
        #         ollama_pub_key = gr.Button("Ollama Pub Key")    
        save_output = gr.Markdown("---")
        save_button = gr.Button("Save Model", visible=True, interactive=True)
        save_button.click(save_model, inputs=[model_name, hub_model_name, hub_token, gguf_16bit, gguf_8bit, gguf_4bit, gguf_custom, gguf_custom_value, merge_16bit, merge_4bit, just_lora, push_to_hub], outputs=[save_output, save_button])

    with gr.Tab("Inference"):
        with gr.Row():
            input_text = gr.Textbox(label="Input Text", lines=4, value="""\
Continue the fibonnaci sequence.
# instruction
1, 1, 2, 3, 5, 8
# input
""", interactive=True)
            output_text = gr.Textbox(label="Output Text", lines=4, value="", interactive=False)

        inference_button = gr.Button("Inference", visible=True, interactive=True)
        inference_button.click(inference, inputs=[input_text], outputs=[output_text, inference_button])
    load_btn.click(load_model, inputs=[initial_model_name, load_in_4bit, max_sequence_length, hub_token], outputs=[output, load_btn, train_btn, initial_model_name, load_in_4bit, max_sequence_length])

demo.launch()
