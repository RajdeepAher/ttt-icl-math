from huggingface_hub import login
from dotenv import load_dotenv
import os
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

class KNNLanguageModel:
    def __init__(self,
                 model_name='meta-llama/Llama-3.1-8B-Instruct',
                 retrieval_dataset_name='dyyyyyyyy/ScaleQuest-Math',
                 eval_dataset_name='HuggingFaceH4/MATH-500',
                 embedding_model='Snowflake/snowflake-arctic-embed-l',
                 k=3):
        """
        Args:
            model_name (str): Name of the language model
            eval_dataset_name (str): Dataset for evaluation
            retrieval_dataset_name (str): Dataset for retrieval corpus
            embedding_model (str): Sentence transformer model for embeddings
            k (int): Number of nearest neighbors to retrieve
        """

        # Setup quantization configuration for 4-bit model
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config,
            device_map='auto'
        )

        # Load both evaluation and retrieval datasets
        self.eval_dataset = load_dataset(eval_dataset_name)['test']
        self.retrieval_dataset = load_dataset(retrieval_dataset_name)['train']

        self.embedding_model = SentenceTransformer(embedding_model)
        self.k = k

        self._prepare_retrieval_index()

    def _prepare_retrieval_index(self):
        print("Preparing FAISS retrieval index...")
        embeddings = []
        self.contexts = []
        self.solutions = []
        
        for sample in tqdm(self.retrieval_dataset):
            try:
                problem = sample.get('query', '')
                solution = sample.get('response', '')
                
                if not problem or not solution:  # Skip if either is empty
                    continue
                    
                # Store the problem and solution for later retrieval
                self.contexts.append(problem)
                self.solutions.append(solution)
                
                # Combine problem and solution for embedding
                combined_text = f"Problem: {problem}\nSolution: {solution}"
                
                # Generate embeddings for the combined text
                embedded = self.embedding_model.encode(combined_text, convert_to_tensor=True)
                embeddings.append(embedded.cpu().numpy())
                
            except Exception as e:
                print(f"Error embedding sample: {e}")
                continue

        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        print("Embedding array shape:", embeddings.shape)

        # Validate embeddings
        if embeddings.size == 0:
            raise ValueError("No embeddings were generated. Check the retrieval dataset and embedding model.")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings should have 2 dimensions, got {embeddings.ndim}")

        # Normalize embeddings for FAISS
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"FAISS index created with {len(embeddings)} embeddings.")


    def retrieve_nearest_neighbors(self, query, k=None):
        """
        Retrieve k nearest neighbors for a given query
        """
        k = k or self.k

        query_embedding = self.embedding_model.encode(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.index.search(query_embedding, k)

        retrieved_contexts = [self.contexts[i] for i in indices[0]]
        retrieved_solutions = [self.solutions[i] for i in indices[0]]

        return retrieved_contexts, retrieved_solutions, similarities[0]

    def generate_with_retrieval(self, query, max_length=4096):
        """
        Generate response with retrieved in-context examples
        """
        retrieved_contexts, retrieved_solutions, similarities = self.retrieve_nearest_neighbors(query)

        prompt ="Output <|eot_id|> at the end of final solution. Use \boxed{} only once in each solution, only for the final answer of the asked question."
        prompt += "Here are some similar math problems and their solutions:\n\n"
        for ctx, sol in zip(retrieved_contexts, retrieved_solutions):
            prompt += f"Problem: {ctx}\nSolution: {sol}\n\n"
        prompt += f"Now solve this problem:\n{query}\nSolution:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        generated_solution = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return generated_solution, similarities

    def evaluate_model(self, num_samples=None):
        """
        Evaluate model on ScaleQuest-Math dataset

        Args:
            num_samples (int, optional): Number of samples to evaluate. If None, uses entire dataset.

        Returns:
            dict: Evaluation results containing predictions and metrics
        """

        results=[]

        eval_data = self.eval_dataset
        if num_samples:
            eval_data = eval_data.select(range(min(num_samples, len(self.eval_dataset))))

        print(f"Evaluating on {len(eval_data)} samples from MATH-500...")
        check =0
        for item in tqdm(eval_data):
            query = item.get('problem', None)
            if not query:
                continue

            try:
                # retrieved_contexts, retrieved_solutions, similarities = self.retrieve_nearest_neighbors(query)
                generated_solution, similarities = self.generate_with_retrieval(query)
                if check ==0:
                    print('QUERY--------------')
                    print(query)
                    print('GENERATED SOLUTION--------------')
                    print(generated_solution)
                    print('GROUND TRUTH--------------')
                    print(item.get('solution', ''))
                    check +=1

                results.append({
                    'queries': query,
                    'predictions': generated_solution,
                    'ground_truth': item.get('solution', ''),
                    'retrieval_similarities': similarities.tolist()
                })

            except Exception as e:
                print(f"Error processing query: {e}")
                continue

        return results


if __name__=='__main__':

    import json
    knn_lm = KNNLanguageModel()
    results=[]
    print("Starting evaluation...")
    results = knn_lm.evaluate_model()
    with open('evaluation_results_knn.json', 'w') as f:
        json.dump(results, f, indent=4)