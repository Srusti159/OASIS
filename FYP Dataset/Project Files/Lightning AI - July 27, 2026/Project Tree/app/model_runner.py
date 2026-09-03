# -----------------------------------------
# IMPORT BACKENDS
# -----------------------------------------

from models.qwen_backend import run_qwen
from models.llama_backend import run_llama
from models.mistral_backend import run_mistral

# -----------------------------------------
# OPTIONAL GEMMA IMPORT
# -----------------------------------------

try:

    from models.gemma_backend import run_gemma

    GEMMA_AVAILABLE = True

except Exception:

    GEMMA_AVAILABLE = False

    print(
        "Gemma backend not available — skipping."
    )

# -----------------------------------------
# RUN ALL MODELS
# -----------------------------------------

# -----------------------------------------
# RUN ALL MODELS
# -----------------------------------------

def run_all_models(prompt, skip_models=None):
    
    if skip_models is None:
        skip_models = []

    results = []

    # -----------------------------------------
    # QWEN
    # -----------------------------------------
    # if "qwen" not in skip_models:
    #     try:
    #         print("\nRunning Qwen...\n")
    #         qwen_result = run_qwen(prompt)
    #         results.append(qwen_result)
    #     except Exception as e:
    #         print(f"Qwen failed: {e}")

    # -----------------------------------------
    # LLAMA
    # -----------------------------------------
    # if "llama" not in skip_models:
    #     try:
    #         print("\nRunning Llama...\n")
    #         llama_result = run_llama(prompt)
    #         results.append(llama_result)
    #     except Exception as e:
    #         print(f"Llama failed: {e}")

    # -----------------------------------------
    # MISTRAL
    # # -----------------------------------------
    if "mistral" not in skip_models:
        try:
            print("\nRunning Mistral...\n")
            mistral_result = run_mistral(prompt)
            results.append(mistral_result)
        except Exception as e:
            print(f"Mistral failed: {e}")

    # -----------------------------------------
    # GEMMA (CONDITIONAL)
    # -----------------------------------------
    if GEMMA_AVAILABLE and "gemma" not in skip_models:
        try:
            print("\nRunning Gemma...\n")
            gemma_result = run_gemma(prompt)
            results.append(gemma_result)
        except Exception as e:
            print(f"Gemma failed: {e}")

    # -----------------------------------------
    # CHECK RESULTS
    # -----------------------------------------
    # Optional: Update this error check so it doesn't crash if ALL models were 
    # skipped but we still have existing summaries being evaluated.
    if not results and not skip_models:
        raise Exception("All models failed!")

    return results