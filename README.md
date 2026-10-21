# ✨ Apsara — AI Ride Assistant

<p align="center">
  <img src="apsara/assets/fictional_model.jpeg" alt="Apsara — an AI-inspired character" width="100%">
</p>

<p align="center">
  <strong>An AI-powered ride assistant that helps passengers find rides and drivers manage scheduled trips.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-AI%20Fine--Tuning-7C3AED?style=for-the-badge" alt="AI Fine-tuning">
  <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B-2563EB?style=for-the-badge" alt="Qwen2.5-1.5B">
  <img src="https://img.shields.io/badge/Domain-Ride%20Sharing-059669?style=for-the-badge" alt="Ride Sharing">
  <img src="https://img.shields.io/badge/Status-In%20Development-F59E0B?style=for-the-badge" alt="In Development">
</p>

## 🌌 About Apsara

**Apsara** is an AI assistant for a ride-sharing application that connects drivers who have already planned a journey with passengers travelling along the same route.

Drivers can schedule upcoming trips, specify their departure time and available seats, and set a passenger contribution amount. Passengers can search for suitable rides, compare available options, and book seats for an affordable journey.

Apsara aims to make this experience easier through natural-language conversations.

The project uses **Qwen2.5-1.5B-Instruct** as its initial base language model. The goal is to specialize the model for ride-sharing conversations and connect it to application APIs so it can retrieve real trip information instead of relying on memorized data.

> Apsara is currently under development. Fine-tuning, API integration, and evaluation are ongoing or planned stages; production readiness has not yet been established.

## 🎯 Project Goals

- 🤖 Develop an AI assistant specialized in ride-sharing.
- 💬 Understand passenger questions and driver requests expressed in natural language.
- 🚗 Help drivers understand how to schedule and manage trips.
- 📍 Help passengers find rides matching their pickup point, destination, and preferred departure time.
- 💺 Retrieve current seat availability and ride details through backend APIs.
- 💰 Help passengers compare available rides and their displayed contribution amounts.
- 🧠 Experiment with parameter-efficient fine-tuning techniques such as LoRA.
- 🔌 Explore controlled API tool calling for ride searches and booking workflows.
- 🧪 Evaluate response quality, factual accuracy, and correct use of application tools.

## 🚘 Core Use Cases

### 1. Driver — Schedule a Trip

A driver planning to travel from Kolenchery to Kochi at 10 AM can schedule the journey in the application.

Apsara should be able to explain the scheduling process and help collect details such as:

- Starting point and destination
- Travel date and departure time
- Available passenger seats
- Passenger contribution amount

The application backend remains responsible for validating and saving the scheduled trip.

### 2. Passenger — Find a Ride

A passenger can ask:

> "Find a ride from Kolenchery to Kochi tomorrow at 10 AM."

Apsara should extract the route and travel-time requirements, call the ride-search API, and present matching trips returned by the application.

### 3. Check Ride Details

Apsara should help passengers retrieve information such as:

- Available seats
- Pickup and drop-off points
- Departure time
- Displayed passenger contribution
- Relevant booking details

These details must come from the backend rather than being invented by the model.

### 4. Booking Assistance

Apsara should help a passenger select a suitable ride, clarify missing information, and request confirmation before initiating a booking.

The backend must validate the trip, seat availability, user authorization, and booking rules before creating a reservation.

### 5. General Application Support

Apsara should answer questions about scheduling trips, searching for rides, bookings, cancellations, and other supported application features using the actual rules defined by the application.

## 🏗️ Intended Architecture

```text
Passenger / Driver
        |
        v
Ride-Sharing Application
        |
        v
Apsara AI Assistant
Qwen2.5-1.5B-Instruct
        |
        v
Intent Understanding
and Tool Selection
        |
        v
Controlled Backend API Calls
        |
        v
Ride Search / Trip Details /
Seat Availability / Booking
        |
        v
Application Database
        |
        v
API Results Returned to Apsara
        |
        v
Natural-Language Response
```

This describes the intended architecture, not a claim that every component has already been implemented.

The language model handles conversation and determines when an application tool is needed. The backend remains the source of truth for live ride data and performs all authorized state-changing operations.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Model development and dataset processing |
| Qwen2.5-1.5B-Instruct | Initial base language model |
| Hugging Face Transformers | Model loading and experimentation |
| Hugging Face PEFT | Planned LoRA fine-tuning experiments |
| Hugging Face Datasets | Dataset loading and processing |
| llama.cpp | Local inference with GGUF models |
| Vulkan | AMD GPU acceleration for supported inference workloads |
| JSONL | Conversational training-data format |
| Git & GitHub | Version control and documentation |

The final fine-tuning configuration, training hardware, and backend integration technologies are still being evaluated or implemented.

## 📚 Dataset Preparation

Apsara requires conversational data relevant to ride-sharing rather than e-commerce customer support.

The project has a new dataset directory:

```text
apsara/ride_sharing_data/
└── train.jsonl
```

The initial training file contains eight manually prepared ride-sharing conversations covering topics such as:

- How the application works
- Driver trip scheduling
- Passenger ride searches
- Comparing prices
- Checking seat availability
- Handling unavailable rides
- Booking assistance
- Cancellation guidance

These examples are only an initial seed dataset. They are not sufficient on their own for reliable fine-tuning or comprehensive evaluation.

### Legacy Dataset Experiment

The project previously used the [Bitext retail e-commerce chatbot dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset) to experiment with conversational data preparation.

The previous preparation workflow included JSONL conversion, dataset inspection, placeholder auditing, filtering, and validation.

The original files remain under `apsara/prepared_data/`. They are retained for reference and are **not intended as the primary training data for the ride-sharing assistant**.

The earlier filtered dataset contained 11,712 examples, but those examples were e-commerce-focused. This figure does not represent the size of the new ride-sharing dataset.

**Dataset licensing:** Review the dataset's license and terms before redistributing its contents or using any resulting model commercially.

## 🧠 Model Development Roadmap

- [x] Select Qwen2.5-1.5B-Instruct as the initial base model.
- [x] Run local inference using a GGUF model and Vulkan.
- [x] Prepare and validate the original e-commerce conversational dataset.
- [x] Create a separate ride-sharing dataset directory.
- [x] Add and validate eight initial ride-sharing conversations.
- [ ] Expand and review the ride-sharing training dataset.
- [ ] Create a separate evaluation dataset.
- [ ] Evaluate the base model on ride-sharing requests.
- [ ] Finalize a memory-conscious fine-tuning environment.
- [ ] Run an initial LoRA training experiment.
- [ ] Evaluate the fine-tuned model on held-out examples.
- [ ] Design structured API tool-calling examples.
- [ ] Integrate controlled ride-search API calls.
- [ ] Implement and test booking workflows with backend validation.
- [ ] Document performance, limitations, and evaluation results.

## 💻 Local Model Inference

A quantized GGUF version of Qwen2.5-1.5B-Instruct has been tested locally using `llama.cpp` with Vulkan acceleration.

The local model file used in the experiment is:

```text
apsara/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Example command:

```bash
llama-cli \
  -m apsara/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --device Vulkan1 \
  --gpu-layers 99 \
  --ctx-size 2048
```

The device name may differ depending on the local `llama.cpp` build.

This command runs the **base model for inference**. It does not perform fine-tuning or connect the model to ride-search APIs.

Model weights are not included in this repository. Obtain model files according to the publisher's instructions and terms.

## 🧪 Training and Evaluation

The project is exploring parameter-efficient fine-tuning, particularly LoRA, to adapt the base model to ride-sharing conversations.

Planned evaluation areas include:

- Correctly identifying ride-search requests
- Extracting routes, dates, and departure times
- Asking for missing trip details
- Avoiding invented ride listings, prices, and seat counts
- Selecting the correct backend tool
- Handling unavailable rides and API errors
- Confirming booking details before state-changing actions

Local GPU inference does not guarantee that GPU fine-tuning is supported. The training framework and hardware configuration must be tested separately.

## 📁 Project Structure

```text
LLM-MODEL-E-COMMERCE/
├── README.md
├── env/
└── apsara/
    ├── assets/
    │   └── fictional_model.jpeg
    ├── models/
    │   └── qwen/
    │       └── qwen2.5-1.5b-instruct-q4_k_m.gguf
    ├── prepared_data/
    │   ├── train.jsonl
    │   ├── test.jsonl
    │   └── filtered_experiment/
    │       └── train.jsonl
    ├── ride_sharing_data/
    │   └── train.jsonl
    ├── prepare_dataset.py
    ├── inspect_dataset.py
    ├── audit_dataset.py
    ├── filter_dataset.py
    ├── validate_filtered.py
    ├── placeholder_audit.py
    ├── download_datasets.py
    └── llm-test.py
```

The structure reflects the current project files and may evolve as dedicated training, API integration, and evaluation scripts are added.

## 🔒 Data and Model Files

Large model weights, generated checkpoints, virtual environments, and local datasets should be excluded from Git tracking unless there is a specific reason to publish them.

This keeps the repository focused on source code, documentation, and reproducible project workflows.

Do not commit API keys, access tokens, database credentials, private user information, or other secrets.

## ⚠️ Current Limitations

- Ride-sharing fine-tuning has not yet been completed.
- The initial ride-sharing dataset contains only eight examples.
- API tool calling and live ride retrieval are planned capabilities, not yet confirmed as implemented.
- Real-time ride availability, pricing, and booking status must come from the backend.
- Local GPU inference support does not guarantee GPU fine-tuning support.
- The model may generate incorrect or incomplete responses and requires evaluation before real-world deployment.
- Booking operations require backend validation, authorization, and appropriate confirmation flows.

## 🤝 Contributions

Suggestions, bug reports, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request describing your changes.

## 📜 Acknowledgements

- [Qwen](https://huggingface.co/Qwen) for the base language model.
- [Hugging Face](https://huggingface.co/) for model and dataset tooling.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local GGUF inference.
- [Bitext](https://huggingface.co/bitext) for the previous e-commerce dataset experiment.

## 👨‍💻 Project

**Apsara — AI Ride Assistant**

An ongoing project exploring specialized language models, ride-sharing assistance, and backend API integration.

---

<p align="center">
  <em>From natural-language requests to smarter shared journeys — one experiment at a time.</em>
</p>
