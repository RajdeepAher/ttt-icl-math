import json
from tqdm import tqdm
import re
from collections import Counter
from datasets import load_dataset
import os

dataset = load_dataset("yobro4619/multihop_rag_balanced_sample")  
def get_gold(query):
    for item in dataset['train']:  
        if item['query'] == query:
            return item['answer']
    return ''

def has_intersection(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0

# Function to extract the answer
def extract_answer(input_string):
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string

def calculate_metrics(pred_list, gold_list):
    tp = sum(1 for pred, gold in zip(pred_list, gold_list) if has_intersection(pred.lower(), gold.lower()))
    fp = sum(1 for pred, gold in zip(pred_list, gold_list) if not has_intersection(pred.lower(), gold.lower()))
    fn = len(gold_list) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# Function to evaluate a single model's output
def evaluate_model(doc_data, model_name):
    type_data = {}
    overall_pred_list = []
    overall_gold_list = []
    
    print(f"\nEvaluating {model_name}:")
    print("-" * 50)
    
    for d in tqdm(doc_data):
        model_answer = d['model_answer']
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)
        gold = get_gold(d['query'])
        if gold:
            question_type = d['question_type']
            if question_type not in type_data:
                type_data[question_type] = {'pred_list': [], 'gold_list': []}
            type_data[question_type]['pred_list'].append(model_answer)
            type_data[question_type]['gold_list'].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)
    
    # Print results for each question type
    for question_type, data in type_data.items():
        precision, recall, f1 = calculate_metrics(data['pred_list'], data['gold_list'])
        print(f"\nQuestion Type: {question_type}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" F1 Score: {f1:.2f}")
    
    # Calculate and print overall metrics
    overall_precision, overall_recall, overall_f1 = calculate_metrics(overall_pred_list, overall_gold_list)
    print(f"\nOverall Metrics for {model_name}:")
    print(f" Precision: {overall_precision:.2f}")
    print(f" Recall: {overall_recall:.2f}")
    print(f" F1 Score: {overall_f1:.2f}")
    
    return {
        'model_name': model_name,
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'type_metrics': type_data
    }

# Main execution
def main():
    # Get all JSON files from qa_output directory
    qa_output_dir = 'qa_output'
    results = []
    
    for filename in os.listdir(qa_output_dir):
        if filename.endswith('.json'):
            model_name = filename[:-5]  # Remove .json extension
            file_path = os.path.join(qa_output_dir, filename)
            
            with open(file_path, 'r') as file:
                doc_data = json.load(file)
            
            # Evaluate the model
            model_results = evaluate_model(doc_data, model_name)
            results.append(model_results)
    
    # Print comparative summary
    print("\n" + "="*50)
    print("Comparative Summary of All Models")
    print("="*50)
    
    # Create a table header
    print(f"{'Model Name':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 56)
    
    # Print each model's metrics
    for result in results:
        metrics = result['overall_metrics']
        print(f"{result['model_name']:<20} {metrics['precision']:.2f}{' '*8} {metrics['recall']:.2f}{' '*8} {metrics['f1']:.2f}")

if __name__ == "__main__":
    main()
