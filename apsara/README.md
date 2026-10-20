# ✨ Apsara — E-commerce AI Assistant

<p align="center">
  <img src="assets/fictional_model.jpeg" alt="Apsara — a celestial AI-inspired character" width="100%">
</p>

<p align="center">
  <strong>Building a specialized AI assistant for e-commerce customer support.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-AI%20Fine--Tuning-7C3AED?style=for-the-badge" alt="AI Fine-tuning">
  <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B-2563EB?style=for-the-badge" alt="Qwen2.5-1.5B">
  <img src="https://img.shields.io/badge/Status-In%20Development-F59E0B?style=for-the-badge" alt="In development">
</p>

## 🌌 About Apsara

**Apsara** is an AI model project focused on building a helpful, professional, and context-aware customer-service assistant for e-commerce applications.

The project starts with the **Qwen2.5-1.5B-Instruct** base model and aims to adapt it to e-commerce customer-support conversations through supervised fine-tuning.

The goal is to help customers get clear answers to common questions about orders, returns, refunds, shipping, and other shopping-related concerns.

> Apsara is currently under development. Fine-tuning and evaluation results will be documented as the project progresses.

## 🎯 Project Goals

- 🤖 Adapt a pretrained language model for e-commerce customer support.
- 💬 Generate helpful, professional responses to customer queries.
- 📦 Prepare and validate conversational training data.
- 🧠 Experiment with parameter-efficient fine-tuning techniques such as LoRA.
- 🧪 Evaluate response quality using questions not seen during training.
- 🚀 Work toward integrating the resulting model into an e-commerce application.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Model development and data processing |
| Qwen2.5-1.5B-Instruct | Base language model |
| Hugging Face Datasets | Dataset loading and processing |
| llama.cpp | Local GGUF model inference |
| Vulkan | AMD GPU acceleration for supported inference workloads |
| Git & GitHub | Version control and project collaboration |

*The final fine-tuning framework and training hardware are still being evaluated.*

## 📊 Dataset Preparation

The project uses the [Bitext retail e-commerce chatbot dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset) as its initial source of customer-support examples.

The dataset preparation workflow includes:

1. Loading and inspecting the source dataset.
2. Converting examples into conversational JSONL format.
3. Auditing duplicate questions, long responses, and unresolved placeholders.
4. Filtering training examples containing unresolved `{{...}}` placeholders.
5. Validating the filtered dataset.

### Current preparation results

| Metric | Result |
|---|---:|
| Filtered training examples | 11,712 |
| Invalid JSONL records | 0 |
| Empty message records | 0 |
| Unresolved placeholders in filtered training data | 0 |

These figures describe the current dataset preparation stage, not model performance.

**Dataset licensing:** Review the dataset's license and terms before redistributing its contents or using the resulting model commercially.

## 🧠 Model Development Roadmap

- [x] Select the initial base model.
- [x] Run local inference using a GGUF model and Vulkan.
- [x] Prepare conversational training data.
- [x] Audit and filter unresolved placeholders from the training set.
- [x] Validate the filtered training dataset.
- [ ] Finalize the fine-tuning environment.
- [ ] Run an initial training experiment.
- [ ] Fine-tune the model.
- [ ] Evaluate responses on held-out customer questions.
- [ ] Document limitations and training results.
- [ ] Explore application integration.

## 💻 Local Inference

A GGUF version of the base model was tested locally using `llama.cpp` with Vulkan acceleration.

Example command:

```bash
llama-cli \
  -m models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --device Vulkan1 \
  --gpu-layers 99 \
  --ctx-size 2048
```

This command runs the **base model for inference**. It does not perform fine-tuning.

Model weights are not included in this repository. Follow the model publisher's instructions and terms to obtain them.

## 📁 Project Structure

```text
apsara/
├── assets/
│   └── apsara-banner.png
├── prepare_dataset.py
├── inspect_dataset.py
├── audit_dataset.py
├── filter_dataset.py
├── validate_filtered.py
├── placeholder_report.py
├── requirements.txt
├── .gitignore
└── README.md
```

The structure will evolve as training scripts and evaluation tools are added.

## 🔒 Data and Model Files

Large model weights, generated checkpoints, virtual environments, and local datasets are excluded from Git tracking.

This keeps the repository focused on source code, documentation, and reproducible project workflows.

## 🤝 Contributions

Suggestions, bug reports, and improvements are welcome. If you want to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request describing your changes.

## ⚠️ Current Limitations

- Apsara has not yet completed fine-tuning.
- Response quality and benchmark results have not yet been established.
- Local GPU inference support does not guarantee GPU fine-tuning support.
- The model may produce incorrect or incomplete answers and should be evaluated before real customer use.

## 📜 Acknowledgements

- [Qwen](https://huggingface.co/Qwen) for the base model.
- [Bitext](https://huggingface.co/bitext) for the initial e-commerce conversational dataset.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local GGUF inference.

## 👨‍💻 Project

**Apsara — E-commerce AI Assistant**

An ongoing learning and development project exploring language-model fine-tuning for customer support.

---

<p align="center">
  <em>From a pretrained model to a specialized assistant — one experiment at a time.</em>
</p>