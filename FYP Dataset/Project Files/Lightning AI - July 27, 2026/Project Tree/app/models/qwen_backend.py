# =========================================
# FILE: app/models/qwen_backend.py
# =========================================

import gc
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def run_qwen(prompt):

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True
    )

    print("\nLoading Qwen...\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    start_time = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512
        )

    end_time = time.time()

    generated_tokens_only = outputs[0][
        inputs.input_ids.shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens_only,
        skip_special_tokens=True
    )

    inference_time = end_time - start_time

    generated_tokens = (
        outputs.shape[1]
        - inputs.input_ids.shape[1]
    )

    tokens_per_second = (
        generated_tokens / inference_time
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    del model
    del tokenizer
    del inputs
    del outputs

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": "Qwen2.5-7B",
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