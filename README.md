# O.A.S.I.S: Open-Source Audio-Visual Summarisation and Insight System

> **O.A.S.I.S** is an end-to-end multimodal meeting intelligence and benchmarking framework. It synchronizes long-form multi-speaker conversational audio (`.wav`) with visual slide presentations (`.pdf`), transcribes and diarizes speakers, parses layout-aware visual slide text, fuses both streams into a unified chronological context, and benchmarks multiple state-of-the-art open-source Large Language Models (LLMs) under strict VRAM constraints using an automated 6-metric evaluation suite.

---

## 📌 Architecture Overview

```
                                  ┌──────────────────────────┐
                                  │  Input: Audio (.wav) &   │
                                  │  Slide Deck (.pdf)       │
                                  └────────────┬─────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         [Part 1: Audio Engine]                              [Part 2: PDF Engine]
   • Whisper large-v3 Transcription                    • PDF Rasterization (300 DPI)
   • Phoneme Alignment (wav2vec2 CTC)                  • Moondream2 VLM Vision Analysis
   • Pyannote 3.1 Speaker Diarization                  • Structural Markdown Extraction
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                   [Part 3: Fusion Engine]
                       • Temporal & Structural Concatenation
                       • 120,000 Character Context Budget Clamping
                                               │
                                               ▼
                                   [Part 4: LLM Engine]
                       • Sequential Execution on Single 40GB A100 GPU
                       • Models: GLM-4 9B | Llama 3.1 8B | Mistral 7B | Qwen 2.5 7B
                       • 3-Step VRAM Annihilation Protocol
                                               │
                                               ▼
                                  [Part 5: Evaluator Engine]
                       • ROUGE-1 / ROUGE-2 / ROUGE-L (Lexical)
                       • BERTScore (RoBERTa-large Contextual)
                       • BARTScore (Log-Likelihood Generative)
                       • DeBERTa-v3 NLI (Hallucination Detection)
                       • LLM-as-a-Judge (Factual Grounding)
                                               │
                                               ▼
                                   [Part 6: Report Generator]
                       • Telemetry Tracking (HardwareSpy NVML)
                       • Comprehensive Markdown Benchmark Matrix
```

---

## ✨ Key Features

- **High-Precision Multi-Speaker Audio Pipeline**:
  - Employs **Whisper large-v3** for low Word Error Rate (WER) transcription.
  - Leverages **WhisperX** forced alignment via language-specific `wav2vec2` CTC models for millisecond word timestamps.
  - Attributed speaker turns via **Pyannote 3.1** agglomerative voice clustering.
- **Multimodal Visual Context Extraction**:
  - Uses **Moondream2** (1.8B VLM) to parse slides, understanding layout hierarchies, text flow, diagrams, and figures without relying on brittle traditional OCR engines.
- **Strict VRAM Annihilation Protocol**:
  - Executes multiple 7B–9B parameter LLMs sequentially on a **single 40GB GPU**.
  - Releases GPU memory completely between inference cycles via:
    1. Python reference destruction (`del model, tokenizer`).
    2. Explicit garbage collection (`gc.collect()`).
    3. CUDA caching allocator flush (`torch.cuda.empty_cache()`).
- **Real-Time Hardware Telemetry**:
  - Integrated `HardwareSpy` continuously monitors GPU memory utilization, peak VRAM allocations, and inference latencies via `pynvml` across daemon polling threads.
- **Automated 6-Metric Evaluation Suite**:
  - Assesses summaries against ground-truth references using Lexical (ROUGE), Semantic (BERTScore), Generative (BARTScore), Factual Entailment (DeBERTa-v3 NLI), and LLM-as-a-Judge protocols.

---

## 🛠️ Models Benchmark Suite

| Pipeline Stage | Model Identifier | Precision | Parameter Count |
|---|---|---|---|
| **Transcription** | `openai/whisper-large-v3` | Float16 | 1.55B |
| **Alignment** | `wav2vec2` (Language-specific CTC) | Float16 | ~317M |
| **Diarization** | `pyannote/speaker-diarization-3.1` | Float32 | ~50M |
| **Vision OCR / VLM** | `vikhyatk/moondream2` | BFloat16 | 1.86B |
| **LLM Evaluation** | `THUDM/glm-4-9b-chat` | BFloat16 | 9B |
| **LLM Evaluation** | `meta-llama/Llama-3.1-8B-Instruct` | BFloat16 | 8B |
| **LLM Evaluation** | `mistralai/Mistral-7B-Instruct-v0.3` | BFloat16 | 7B |
| **LLM Evaluation** | `Qwen/Qwen2.5-7B-Instruct` | BFloat16 | 7B |
| **Hallucination NLI** | `cross-encoder/nli-deberta-v3-base` | Float32 | 86M |
| **Evaluation Likelihood** | `facebook/bart-large-cnn` | Float32 | 406M |

---

## 📂 Repository Structure

```text
oasis/
├── config.py             # Central frozen configuration, paths, prompts, hyperparameters
├── audio_engine.py       # Part 1: WhisperX transcription, alignment, and Pyannote diarization
├── pdf_engine.py         # Part 2: PDF rendering (pdf2image) & Moondream2 VLM extraction
├── fusion_engine.py      # Part 3: Text alignment, context structuring, and token truncation
├── llm_engine.py         # Part 4: Sequential LLM summarization with VRAM annihilation
├── evaluator.py          # Part 5: 6-metric automated benchmark scoring (ROUGE, BERT, BART, NLI)
├── telemetry.py          # Background NVML GPU polling and hardware telemetry (HardwareSpy)
├── main.py               # Master orchestration script across dataset indices
├── requirements.txt      # System dependencies and exact framework versions
└── .env.example          # HuggingFace token and environment variable templates
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Linux / WSL2 environment with CUDA 12.1+
- NVIDIA GPU with $\ge$ 24GB VRAM 
- Python 3.10 or 3.11

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/oasis.git
cd oasis

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install system dependencies (Poppler for PDF rasterization, libsndfile for audio)
sudo apt-get update && sudo apt-get install -y poppler-utils libsndfile1 ffmpeg

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and provide your gated model access token:
```bash
cp .env.example .env
```
Inside `.env`:
```env
HF_TOKEN=your_huggingface_access_token_here
```
> **Note**: Your token must have accepted the user agreements for `meta-llama/Llama-3.1-8B-Instruct` and `pyannote/speaker-diarization-3.1` on Hugging Face.

### 4. Data Preparation
Place meeting audio and presentation files under the `./dataset/` directory structured by index:
```text
dataset/
├── Audios/
│   ├── AMI_01.wav
│   └── AMI_02.wav
├── PDFs/
│   ├── AMI_01.pdf
│   └── AMI_02.pdf
└── GroundTruth/
    ├── AMI_01.txt
    └── AMI_02.txt
```

### 5. Running the Pipeline
Execute the master orchestration pipeline:
```bash
python main.py
```
To run for a specific meeting index, configure `config.py` or specify target files inside `main.py`.

---

## 📊 Benchmark Evaluation Metrics

OASIS evaluates candidate summaries across multiple orthogonal dimensions:

1. **Lexical Overlap**:
   - **ROUGE-1, ROUGE-2, ROUGE-L**: Measures n-gram recall and longest common subsequence matches against human reference summaries.
2. **Contextual Semantic Similarity**:
   - **BERTScore (RoBERTa-large)**: Computes cosine similarity between token embeddings, rewarding semantic meaning over surface tokens.
3. **Information Density & Likelihood**:
   - **BARTScore (`facebook/bart-large-cnn`)**: Evaluates the negative log-likelihood of the summary conditioned on the source transcript.
4. **Factual Consistency & Hallucination**:
   - **DeBERTa-v3 Cross-Encoder NLI**: Classifies summary sentences against source premises. The `contradiction` probability serves as the quantitative **Hallucination Rate**.
5. **LLM-as-a-Judge**:
   - Scores quality, conciseness, and coverage on a 1–10 scale using an instruction-tuned judge with few-shot evaluation prompts.

---

## 💡 System Design Highlights

### VRAM Discipline Pattern
```python
def annihilate_model(model, tokenizer):
    """Guarantees complete memory deallocation from CUDA device."""
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
```
This pattern prevents memory fragmentation, enabling deep evaluation runs over multiple LLMs without requiring server reboots or multi-GPU clusters.

---


---

## 🤝 Acknowledgments
- [WhisperX](https://github.com/m-bain/whisperX) for phoneme-level forced alignment.
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) for state-of-the-art neural diarization.
- [Moondream](https://github.com/vikhyat/moondream) for efficient edge visual reasoning.
- [Hugging Face](https://huggingface.co/) for open-source LLM hosting and tooling.
