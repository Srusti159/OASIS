import os
import gc
import re
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress PyTorch Inductor autotune warnings on T4
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"


def clean_multimodal_prompt(prompt):
    """
    Deduplicates repeated visual context bullet points within each time window.
    Reduces prompt size by ~50% while preserving all core visual info.
    """
    # Split by Time Window sections
    sections = prompt.split("---------------------")
    cleaned_sections = []

    for section in sections:
        if "Visual Context" in section:
            # Extract visual context block
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
                    # Only keep unique visual descriptions
                    clean_line = line.strip()
                    if clean_line not in unique_visuals:
                        unique_visuals.append(line)
                else:
                    audio_lines.append(line)

            # Reconstruct section
            cleaned_section = "\n".join(unique_visuals + audio_lines)
            cleaned_sections.append(cleaned_section)
        else:
            cleaned_sections.append(section)

    return "---------------------\n".join(cleaned_sections)


def run_gemma(prompt):

    # 1. Clear VRAM before starting
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("\nLoading Gemma-2-9B (T4 Optimized)...\n")

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Note: Unsloth models are pre-quantized; no custom quantization_config needed
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        attn_implementation="sdpa"  # Memory-efficient attention for T4
    )

    # 3. Clean redundant visual lines from prompt
    cleaned_prompt = clean_multimodal_prompt(prompt)

    # 4. Wrap prompt in Gemma's official chat template
    system_instruction = (
        "You are an expert meeting assistant. Analyze the provided audio transcript "
        "and visual scene descriptions. Provide a concise, structured meeting summary "
        "covering: 1. Main Discussion Points, 2. Key Ideas/Proposals, and 3. Action Items."
    )

    messages = [
        {"role": "user", "content": f"{system_instruction}\n\n{cleaned_prompt}"}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5. Tokenize with strict 8,192 token truncation guard
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to("cuda")

    input_length = inputs.input_ids.shape[1]
    print(f"Gemma Input Tokens (after cleaning): {input_length}")

    # 6. Generate Summary
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding for consistent benchmarking
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()

    # 7. Decode generated tokens only
    generated_tokens_only = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens_only, skip_special_tokens=True).strip()

    # 8. Calculate Efficiency Metrics
    inference_time = end_time - start_time
    generated_tokens = outputs.shape[1] - input_length
    tokens_per_second = generated_tokens / inference_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 9. Clean up VRAM for the next model in the pipeline
    del model
    del tokenizer
    del inputs
    del outputs

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": "Gemma-2-9B",
        "summary": response,
        "latency_sec": round(inference_time, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "peak_vram_mb": round(peak_vram, 2)
    }