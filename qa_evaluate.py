import json
from tqdm import tqdm
import re
import pandas as pd
import os
from collections import Counter

def normalize_answer(answer):
    """Normalize answer by converting to lowercase, removing extra spaces"""
    if not answer:
        return ''
    # Handle No Evidence/Insufficient Information cases
    if answer.lower() in ['no evidence', 'insufficient information', 'insufficient information.']:
        return 'insufficient information'
    return ' '.join(answer.lower().strip().split())

def is_partial_match(pred, gold):
    """Check if prediction is a partial match with gold answer"""
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    
    # Handle No Evidence/Insufficient Information cases
    if pred == 'insufficient information' and gold == 'insufficient information':
        return True
        
    # Split into words
    pred_words = pred.split()
    gold_words = gold.split()
    
    # If prediction is a substring of gold or vice versa
    if pred in gold or gold in pred:
        return True
        
    # Check if last name matches full name
    if len(pred_words) == 1 and len(gold_words) > 1:
        if pred_words[0] == gold_words[-1]:  # Last name match
            return True
    
    # Check if first word matches (for cases like "Caesars" vs "Caesars Sportsbook")
    if len(pred_words) >= 1 and len(gold_words) >= 1:
        if pred_words[0] == gold_words[0]:
            return True
    
    return False

def has_intersection(a, b):
    """Check word intersection between two strings with normalization"""
    a = normalize_answer(a)
    b = normalize_answer(b)
    
    # Handle No Evidence/Insufficient Information cases
    if a == 'insufficient information' and b == 'insufficient information':
        return True
        
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0

def extract_answer(input_string):
    """Extract answer from formatted string"""
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string

def calculate_metrics(pred_list, gold_list):
    """Calculate precision, recall, F1, and accuracy using partial matching"""
    if not pred_list or not gold_list:
        return 0, 0, 0, 0
        
    tp = sum(1 for pred, gold in zip(pred_list, gold_list) 
             if has_intersection(pred, gold))
    fp = sum(1 for pred, gold in zip(pred_list, gold_list) 
             if not has_intersection(pred, gold))
    fn = len(gold_list) - tp
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Calculate accuracy using partial matching
    accuracy = sum(1 for pred, gold in zip(pred_list, gold_list) 
                  if is_partial_match(pred, gold)) / len(gold_list)
    
    return precision, recall, f1, accuracy

def evaluate_model(doc_data, model_name):
    type_data = {}
    overall_pred_list = []
    overall_gold_list = []
    
    # Track skipped queries and null queries
    skipped_count = 0
    null_query_count = 0
    
    print(f"\nEvaluating {model_name}:")
    print("-" * 50)
    
    for d in tqdm(doc_data):
        model_answer = d['model_answer']
        question_type = d['question_type']
        
        # Skip queries with API Error
        if model_answer == "API Error":
            skipped_count += 1
            continue
            
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)
        
        gold = d['gold_answer']  # Use gold_answer directly from the JSON
        
        if question_type not in type_data:
            type_data[question_type] = {'pred_list': [], 'gold_list': []}
        
        type_data[question_type]['pred_list'].append(model_answer)
        type_data[question_type]['gold_list'].append(gold)
        
        # Only add to overall metrics if not null_query
        if question_type != 'null_query':
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)
        else:
            null_query_count += 1
    
    # Print counts
    print(f"Skipped {skipped_count} queries due to API Error")
    print(f"Excluded {null_query_count} null queries from overall metrics")
    print(f"Evaluated {len(overall_pred_list)} queries for overall metrics\n")
    
    # Print results for each question type
    for question_type, data in type_data.items():
        precision, recall, f1, accuracy = calculate_metrics(
            data['pred_list'], data['gold_list']
        )
        print(f"\nQuestion Type: {question_type}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" F1 Score: {f1:.2f}")
        print(f" Accuracy: {accuracy:.2f}")
        print(f" Number of queries: {len(data['pred_list'])}")
    
    # Calculate overall metrics (excluding null_query)
    overall_precision, overall_recall, overall_f1, overall_accuracy = calculate_metrics(
        overall_pred_list, overall_gold_list
    )
    
    return {
        'model_name': model_name,
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'accuracy': overall_accuracy
        },
        'type_metrics': type_data,
        'counts': {
            'skipped': skipped_count,
            'null_query': null_query_count,
            'evaluated': len(overall_pred_list)
        }
    }
def filter_json_by_indices(input_json, use_sampling=True):
    sample_indices_path = "sampled_indices.pkl"
    
    # Load sampled indices if available
    if os.path.exists(sample_indices_path):
        sampled_indices = pd.read_pickle(sample_indices_path)
        print("Loaded precomputed indices.")
    else:
        sampled_indices = None  # Allows function to run without sampling if needed
    
    # Convert JSON input to DataFrame
    file_df = pd.DataFrame(input_json)
    
    # Exclude 'null_query' rows
    file_df = file_df[file_df["question_type"] != "null_query"].reset_index(drop=True)
    
    # Apply sampling only if use_sampling=True and sampled indices exist
    if use_sampling and sampled_indices is not None:
        file_df = file_df.loc[sampled_indices].reset_index(drop=True)
    
    # Convert back to JSON
    return file_df.to_dict(orient="records")


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
            filtered_json = filter_json_by_indices(doc_data)
            # Evaluate the model
            model_results = evaluate_model(filtered_json, model_name)
            results.append(model_results)
    
    qa_output_dir = 'qa_output_mm'
    for filename in os.listdir(qa_output_dir):
        if filename.endswith('.json'):
            model_name = filename[:-5]  # Remove .json extension
            file_path = os.path.join(qa_output_dir, filename)
            
            with open(file_path, 'r') as file:
                doc_data = json.load(file)
            #filtered_json = filter_json_by_indices(doc_data)
            # Evaluate the model
            model_results = evaluate_model(doc_data, model_name)
            results.append(model_results)
    
    # Print comparative summary
    print("\n" + "="*65)
    print("Comparative Summary of All Models")
    print("="*65)
    
    # Create a table header
    print(f"{'Model Name':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Accuracy':<12}")
    print("-" * 68)
    
    # Print each model's metrics
    for result in results:
        metrics = result['overall_metrics']
        print(f"{result['model_name']:<20} {metrics['precision']:.2f}{' '*8} {metrics['recall']:.2f}{' '*8} {metrics['f1']:.2f}{' '*8} {metrics['accuracy']:.2f}")

if __name__ == "__main__":
    main()