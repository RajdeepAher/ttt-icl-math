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
from activeft.sift import Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


load_dotenv()
api_key = os.getenv("HF_API_KEY")
if api_key:
    login(api_key)
    print("Logged in successfully.")
else:
    print("API key not found. Please check your .env file.")

class SIFTModel:
    def __init__(
        self,
    model_name = 'meta-llama/Llama-3.1-8B-Instruct',
    retrieval_dataset_name='dyyyyyyyy/ScaleQuest-Math',
    eval_dataset_name='HuggingFaceH4/MATH-500',
    embedding_model_name='Snowflake/snowflake-arctic-embed-l',
    k=3

    ):
        self.model_name = model_name
        self.eval_dataset_name = eval_dataset_name
        self.retrieval_dataset_name = retrieval_dataset_name
        self.embedding_model_name = embedding_model_name
        self.k = k

        if not hasattr(self, 'quantization_config'):
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        if not hasattr(self, 'model'):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map="auto",
            )

        if not hasattr(self, 'eval_dataset'):
            self.eval_dataset = load_dataset(self.eval_dataset_name)['test']

        if not hasattr(self, 'retrieval_dataset'):
            self.retrieval_dataset = load_dataset(self.retrieval_dataset_name)['train']

        if not hasattr(self, 'embedding_model'):
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self._prepare_retrieval_index()


    def _prepare_retrieval_index(self):
        print("Preparing FAISS retrieval index...")
        embeddings = []
        self.contexts = []
        self.solutions = []

        for sample in tqdm(self.retrieval_dataset):
            try:
                problem = sample.get("query", "")
                solution = sample.get("response", "")
                if not problem or not solution:
                    print('Either problem or solution was missing')
                    continue

                self.contexts.append(problem)
                self.solutions.append(solution)

                combined_text = f"Problem: {problem} \n Solution: {solution}"
                embedded = self.embedding_model.encode(combined_text, convert_to_tensor=True)
                embeddings.append(embedded.cpu().numpy())

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        # Stack embeddings into a single NumPy array
        embeddings = np.vstack(embeddings).astype('float32')
        print("Embedding array shape:", embeddings.shape)

        print("Normalizing embeddings.....")
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)
        print(f"FAISS index created with {len(embeddings)} embeddings.")

        self.retriever = Retriever(
            index=faiss_index,
            llambda=0.02,
            fast=True,
            only_faiss=False
        )



    def retrieve_nearest_neighbors(self, query, k=3):
        """
        Retrieve k nearest neighbors for a given query.

        Args:
            query (str): Input problem to find similar examples.
            k (int, optional): Number of neighbors to retrieve. Defaults to self.k.

        Returns:
            list: Retrieved contexts and solutions.
        """
        k = k or self.k

        # Embed query and reshape for SIFT
        query_embedding = self.embedding_model.encode(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)  # Ensure it's a 2D array of shape (1, d)
        print(query_embedding.shape)  # Shape should be (1, d)
        faiss.normalize_L2(query_embedding)
        # Search with SIFT
        result = self.retriever.search(query_embedding, N=k, K=None)

    # Inspect the return value to check how many elements it returns
        #print(f"For Search result: {result}")
        D, I, V, retrieval_time = result
        print(' For the above result set of indexes are', I)

        # Retrieve contexts and solutions
        retrieved_contexts = [self.contexts[i] for i in I]
        retrieved_solutions = [self.solutions[i] for i in I]

        return retrieved_contexts, retrieved_solutions


    def generate_with_retrieval(self, query, max_length=4096):
        """
        Generate response with retrieved in-context examples
        """
        retrieved_contexts, retrieved_solutions = self.retrieve_nearest_neighbors(query)
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
        return generated_solution

    def evaluate_model(self, num_samples=None):
        """
        Evaluate model on ScaleQuest-Math dataset

        Args:
            num_samples (int, optional): Number of samples to evaluate. If None, uses entire dataset.

        Returns:
            dict: Evaluation results containing predictions and metrics
        """
        self.results =[]
        #  {
        #     'queries': [],
        #     'predictions': [],
        #     'ground_truth': [],
        #     #'retrieval_similarities': []
        # }

        eval_data = self.eval_dataset
        if num_samples:
            eval_data = eval_data.select(range(min(num_samples, len(self.eval_dataset))))

        check = 0
        print(f"Evaluating on {len(eval_data)} samples from MATH-500...")
        for item in tqdm(eval_data):
            query = item.get('problem', None)
            # print(query)
            if not query:
                continue

            try:

                generated_solution = self.generate_with_retrieval(query)
                if check ==0:
                    print('QUERY--------------')
                    print(query)
                    print('GENERATED SOLUTION--------------')
                    print(generated_solution)
                    print('GROUND TRUTH--------------')
                    print(item.get('solution', ''))
                    check +=1

                self.results.append({
                    'query': query,
                    'prediction': generated_solution,
                    'ground_truth': item.get('solution', ''),
                })


            except Exception as e:
                print(f"Error processing query: {e}")
                continue

        return self.results

if __name__='__main__':

    import json
    sift_model = SIFTModel()
    results=[]
    print("Starting evaluation...")
    results = sift_model.evaluate_model(num_samples=10)
    with open('evaluation_results_sift.json', 'w') as f:
    json.dump(results, f, indent=4)