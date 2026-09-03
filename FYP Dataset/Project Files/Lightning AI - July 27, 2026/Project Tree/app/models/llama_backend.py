import gc
import os
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# -----------------------------------------
# MODEL VARIANTS
# -----------------------------------------

MODEL_GPU = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MODEL_CPU = "NousResearch/Meta-Llama-3-8B-Instruct"

# -----------------------------------------
# LIMIT VISUAL CONTEXT TO PREVENT
# PATTERN-CONTINUATION
# -----------------------------------------

def limit_visual_context(prompt, max_descriptions=10):
    """
    Llama continues generating visual descriptions
    if too many are in the prompt. This limits them
    so the model focuses on summarizing instead.
    """

    vis_marker = "VISUAL CONTEXT"
    trans_marker = "TRANSCRIPT"

    vis_idx = prompt.find(vis_marker)
    trans_idx = prompt.find(trans_marker)

    if vis_idx == -1 or trans_idx == -1:
        return prompt

    before_vis = prompt[:vis_idx]
    vis_section = prompt[vis_idx:trans_idx]
    after_trans = prompt[trans_idx:]

    vis_lines = [
        l for l in vis_section.split('\n')
        if l.strip().startswith('[')
    ]

    limited = '\n'.join(
        vis_lines[:max_descriptions]
    )

    if len(vis_lines) > max_descriptions:
        limited += (
            f"\n(... {len(vis_lines) - max_descriptions}"
            f" more scene descriptions omitted)"
        )

    return (
        before_vis
        + "VISUAL CONTEXT (key scenes):\n"
        + limited
        + "\n\n"
        + after_trans
    )


def run_llama(prompt):

    gc.collect()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # =========================================
    # LOAD MODEL (GPU or CPU)
    # =========================================

    if DEVICE == "cuda":

        print("\nLoading Llama (4-bit GPU)...\n")

        model_name = MODEL_GPU

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    else:

        try:
            import psutil
            free_gb = (
                psutil.virtual_memory().available
                / (1024**3)
            )
        except ImportError:
            free_gb = (
                os.sysconf('SC_PAGE_SIZE')
                * os.sysconf('SC_PHYS_PAGES')
                / (1024**3)
            )

        if free_gb < 18:
            raise RuntimeError(
                f"Not enough RAM for Llama on CPU. "
                f"Available: {free_gb:.1f}GB, "
                f"Need: ~18GB. Use GPU instead."
            )

        print(
            f"\nLoading Llama (CPU, "
            f"{free_gb:.1f}GB free)...\n"
        )

        model_name = MODEL_CPU

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # =========================================
    # PREPARE PROMPT
    # =========================================

    # limit visual context to prevent the model
    # from continuing the description pattern
    processed_prompt = limit_visual_context(
        prompt,
        max_descriptions=10
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a meeting summary generator. "
                "Output ONLY a professional meeting summary. "
                "NEVER output visual descriptions, timestamps, "
                "or transcript lines. "
                "Start your response with 'Meeting Summary:'"
            )
        },
        {
            "role": "user",
            "content": processed_prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # append output primer to steer generation
    formatted_prompt += "Meeting Summary:\n\n"

    # -----------------------------------------
    # TOKENIZE WITH TRUNCATION
    # -----------------------------------------

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(DEVICE)

    input_length = inputs.input_ids.shape[1]

    print(
        f"Llama input tokens: {input_length}"
    )

    # -----------------------------------------
    # GENERATE
    # -----------------------------------------

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()

    # -----------------------------------------
    # DECODE ONLY GENERATED TOKENS
    # -----------------------------------------

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    # -----------------------------------------
    # METRICS
    # -----------------------------------------

    inference_time = end_time - start_time

    generated_tokens = (
        outputs.shape[1] - input_length
    )

    tokens_per_second = (
        generated_tokens / inference_time
    )

    if torch.cuda.is_available():
        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
    else:
        peak_vram = 0.0

    # -----------------------------------------
    # CLEANUP
    # -----------------------------------------

    del model
    del tokenizer
    del inputs
    del outputs

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": "Llama-3-8B",
        "summary": response,
        "latency_sec": round(
            inference_time,
            2
        ),
        "tokens_per_second": round(
            tokens_per_second,
            2
        ),
        "peak_vram_mb": round(
            peak_vram,
            2
        )
    }
