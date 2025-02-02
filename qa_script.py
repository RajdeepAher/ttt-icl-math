import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
from typing import Any, Generator, List, Optional
from pathlib import Path

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

def transform_json(input_json):
    if isinstance(input_json, list):
        return [transform_json(item) for item in input_json]
    new_retrieval_list = []
    for item in input_json["retrieval_list"]:
        new_text = (
            f"[Excerpt from document]\n"
            f"title: {item['metadata']['title']}\n"
            f"published_at: {item['metadata']['published_at']}\n"
            f"source: {item['metadata']['source']}\n"
            f"Excerpt:\n"
            f"-----\n"
            f"{item['text']}\n"
            f"-----"
        )
        new_retrieval_list.append({
            "text": new_text,
            "score": item["score"]
        })

    new_gold_list = []
    for item in input_json["gold_list"]:
        new_gold_list.append({
            "title": item["title"],
            "author": item["author"],
            "url": item["url"],
            "source": item["source"],
            "category": item["category"],
            "published_at": item["published_at"],
            "fact": item["fact"]
        })

    new_json = {
        "query": input_json["query"],
        "answer": input_json["answer"],
        "question_type": input_json["question_type"],
        "retrieval_list": new_retrieval_list,
        "gold_list": new_gold_list
    }

    return new_json

def query_bot(
        messages,
        model,
        tokenizer,
        temperature=0.01,
        max_new_tokens=512,
        **kwargs,
):
    prefix = "Below is a question followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation."
    
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
        context = '--------------'.join(e['text'] for e in retrieval_list)
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
    
    # Parse arguments
    args = parser.parse_args()

    # Set up CUDA and model settings
    torch.set_default_dtype(torch.float16)
    
    # Login to Hugging Face
    login(token=args.hf_token)

    print("Loading model and tokenizer...")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        device_map="auto"
    )
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
