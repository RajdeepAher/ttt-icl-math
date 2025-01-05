from huggingface_hub import login
from dotenv import load_dotenv
import os
import json
import torch
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()
api_key = os.getenv("HF_API_KEY")
if api_key:
    login(api_key)
    print("Logged in successfully.")
else:
    print("API key not found. Please check your .env file.")


def main():
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    # num_samples = 10
    results = []
    eval_dataset_name='HuggingFaceH4/MATH-500'
    quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eval_data = load_dataset(eval_dataset_name)['test']
    # if num_samples:
    #     eval_data = eval_data.select(range(min(num_samples, len(eval_data))))

    # print(f"Evaluating on {len(eval_data)} samples from MATH-500...")
    # check =0
    # for item in tqdm(eval_data):
    #     query = item.get('problem', None)
    num_evaluated = 0
    check=0
    with open(self.eval_dataset, 'r') as file:
        for line in tqdm(file):
            if num_evaluated == num_samples:
                break
            num_evaluated += 1

            item = json.loads(line.strip())  # Parse each line as a JSON object
            query = item.get("problem", "")

            prompt ="Output <|eot_id|> at the end of final solution. Use \\boxed{} only once in each solution, only for the final answer of the asked question."
            prompt += f"Now solve this problem:\n{query}\nSolution:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=4096,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                temperature =1e-5,
                do_sample=False,
            )


            generated_solution =tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            if check ==0:
                print('QUERY--------------')
                print(query)
                print('GENERATED SOLUTION--------------')
                print(generated_solution)
                print('GROUND TRUTH--------------')
                print(item.get('solution', ''))
                check +=1
            results.append({
                            'query': query,
                            'prediction': generated_solution,
                            'ground_truth': item.get('solution', ''),
                        })
    with open('evaluation_results_vanilla.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__=='__main__':
    main()