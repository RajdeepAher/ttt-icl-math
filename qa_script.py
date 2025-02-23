import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os
from typing import Any, Generator, List, Optional
from pathlib import Path
import pandas as pd

# Create necessary directories
os.makedirs('input_data', exist_ok=True)
os.makedirs('qa_output', exist_ok=True)

def save_list_to_json(lst, filename):
    """
    Save a list of dictionaries to a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(lst, file, indent=4)

# def transform_json(input_json):
#     if isinstance(input_json, list):
#         return [transform_json(item) for item in input_json]
#     new_retrieval_list = []
#     for item in input_json["retrieval_list"]:
#         new_text = (
#             f"[Excerpt from document]\n"
#             f"title: {item['metadata']['title']}\n"
#             # f"published_at: {item['metadata']['published_at']}\n"
#             # f"source: {item['metadata']['source']}\n"
#             f"Excerpt:\n"
#             f"-----\n"
#             f"{item['text']}\n"
#             f"-----"
#         )
#         new_retrieval_list.append({
#             "text": new_text,
#             "score": item["score"]
#         })

#     new_gold_list = []
#     for item in input_json["gold_list"]:
#         new_gold_list.append({
#             "title": item["title"],
#             "author": item["author"],
#             "url": item["url"],
#             "source": item["source"],
#             "category": item["category"],
#             "published_at": item["published_at"],
#             "fact": item["fact"]
#         })

#     new_json = {
#         "query": input_json["query"],
#         "answer": input_json["answer"],
#         "question_type": input_json["question_type"],
#         "retrieval_list": new_retrieval_list,
#         "gold_list": new_gold_list
#     }

#     return new_json
sample_indices_path = "sampled_indices.pkl"

if os.path.exists(sample_indices_path):
    sampled_indices = pd.read_pickle(sample_indices_path)
    print("Loaded precomputed indices.")
else:
    sampled_indices = None  # Allows function to run without sampling if needed

def transform_json(input_json, use_sampling=True):
    # Convert JSON input to DataFrame
    file_df = pd.DataFrame(input_json)

    # Exclude 'null_query' rows
    file_df = file_df[file_df["question_type"] != "null_query"].reset_index(drop=True)

    # Apply sampling only if use_sampling=True and sampled indices exist
    if use_sampling and sampled_indices is not None:
        file_df = file_df.loc[sampled_indices].reset_index(drop=True)

    transformed_data = []
    
    for row in file_df.to_dict(orient="records"):
        new_retrieval_list = [
            {
                "text": (
                    f"[Excerpt from document]\n"
                    f"title: {item['metadata']['title']}\n"
                    f"Excerpt:\n"
                    f"-----\n"
                    f"{item['text']}\n"
                    f"-----"
                ),
                "score": item["score"]
            }
            for item in row["retrieval_list"]
        ]

        new_gold_list = [
            {
                "title": item["title"],
                "author": item["author"],
                "url": item["url"],
                "source": item["source"],
                "category": item["category"],
                "published_at": item["published_at"],
                "fact": item["fact"]
            }
            for item in row["gold_list"]
        ]

        transformed_data.append({
            "query": row["query"],
            "answer": row["answer"],
            "question_type": row["question_type"],
            "retrieval_list": new_retrieval_list,
            "gold_list": new_gold_list
        })

    return transformed_data

def query_bot(
        messages,
        model,
        tokenizer,
        temperature=0.1,
        max_new_tokens=512,
        **kwargs,
):
    prefix = """You are a precise question-answering assistant. Your task is to answer questions based solely on the provided context. Follow these guidelines:

1. Give direct, concise answers without explanations
2. If the context clearly supports a yes/no answer, respond with just 'Yes' or 'No'
3. For factual questions, provide the specific entity, name, or value
4. If the context doesn't contain enough information to answer confidently, respond with 'Insufficient Information'
5. Base your answer only on the given context, not on external knowledge"""
    
    messages = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": messages}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def process_single_file(input_file: Path, model, tokenizer):
    """
    Process a single JSON file and return results
    """
    print(f"\nProcessing file: {input_file.name}")
    
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {input_file} contains invalid JSON")
        return None

    doc_data = transform_json(data)
    save_list = []
    
    for d in tqdm(doc_data, desc=f"Processing {input_file.name}"):
        retrieval_list = d['retrieval_list']
        context = '\n---------'.join(e['text'] for e in retrieval_list)
        prompt = f"Question:{d['query']}\n\nContext:\n\n{context}"
        response = query_bot(prompt, model, tokenizer)
        
        save = {
            'query': d['query'],
            'prompt': prompt,
            'model_answer': response,
            'gold_answer': d['answer'],
            'question_type': d['question_type']
        }
        save_list.append(save)
    
    return save_list

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process JSON files using Hugging Face model')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face login token')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-70b-chat-hf", help='Hugging Face model name')
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization (default: False)")
    
    # Parse arguments
    args = parser.parse_args()

    # Set up CUDA and model settings
    torch.set_default_dtype(torch.float16)
    
    # Login to Hugging Face
    login(token=args.hf_token)

    print("Loading model and tokenizer...")
    # Configure 4-bit quantization
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype = "auto",
        quantization_config=quantization_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print("Model and tokenizer loaded successfully!")

    # Get all JSON files from input_data directory
    input_dir = Path('input_data')
    json_files = list(input_dir.glob('*.json'))
    
    if not json_files:
        print("No JSON files found in input_data directory")
        return

    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for input_file in json_files:
        save_list = process_single_file(input_file, model, tokenizer)
        
        if save_list:
            # Create output filename based on input filename
            output_filename = f'qa_output/llama_{input_file.stem}_output.json'
            save_list_to_json(save_list, output_filename)
            print(f"Results saved to {output_filename}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
