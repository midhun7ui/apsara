from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
)

print(dataset)
print("\nAvailable columns: ")
print(dataset["train"].column_names)

print("\nFirst Example: ")
print(dataset["train"][0])