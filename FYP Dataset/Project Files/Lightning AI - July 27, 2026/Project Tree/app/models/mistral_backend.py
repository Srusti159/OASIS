import os
import gc
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Suppress PyTorch Inductor autotune warnings on T4
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def clean_multimodal_prompt(prompt):
    """
    Deduplicates repeated visual context bullet points within each time window.
    Reduces prompt size by ~50% while preserving all core visual info.
    """
    sections = prompt.split("---------------------")
    cleaned_sections = []

    for section in sections:
        if "Visual Context" in section:
            lines = section.split("\n")
            unique_visuals = []
            audio_lines = []
            in_visual = False

            for line in lines:
                if "Visual Context" in line:
                    in_visual = True
                    unique_visuals.append(line)
                    continue
                elif "Audio Transcript" in line:
                    in_visual = False

                if in_visual and line.strip().startswith("-"):
                    clean_line = line.strip()
                    if clean_line not in unique_visuals:
                        unique_visuals.append(line)
                else:
                    audio_lines.append(line)

            cleaned_section = "\n".join(unique_visuals + audio_lines)
            cleaned_sections.append(cleaned_section)
        else:
            cleaned_sections.append(section)

    return "---------------------\n".join(cleaned_sections)


def run_mistral(prompt):

    # 1. Clear VRAM before starting
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("\nLoading Mistral-7B v0.3 (T4 Optimized)...\n")

    # 2. Configure 4-bit loading specifically for T4 hardware
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 # T4 requires float16
    )

    # 3. Load Tokenizer & fix Mistral's missing pad token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. Load Model with memory-efficient attention (SDPA)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa" 
    )

    # 5. Clean the prompt of duplicate visuals
    cleaned_prompt = clean_multimodal_prompt(prompt)

    # 6. Apply Mistral's v0.3 Chat Template
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert meeting assistant. Analyze the provided audio transcript "
                "and visual scene descriptions. Provide a concise, structured meeting summary "
                "covering: 1. Main Discussion Points, 2. Key Ideas/Proposals, and 3. Action Items."
            )
        },
        {"role": "user", "content": cleaned_prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 7. Tokenize with an 8,192 safety truncation for the T4's memory cap
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to("cuda")

    input_length = inputs.input_ids.shape[1]
    print(f"Mistral Input Tokens (after cleaning): {input_length}")

    # 8. Generate Summary
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False, # Greedy decoding for exact reproducible answers
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()

    # 9. Decode and Extract Metrics
    generated_tokens_only = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens_only, skip_special_tokens=True).strip()

    inference_time = end_time - start_time
    generated_tokens = outputs.shape[1] - input_length
    tokens_per_second = generated_tokens / inference_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 10. Clean up VRAM for the next model
    del model
    del tokenizer
    del inputs
    del outputs

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": "Mistral-7B-v0.3",
        "summary": response,
        "latency_sec": round(inference_time, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "peak_vram_mb": round(peak_vram, 2)
    }