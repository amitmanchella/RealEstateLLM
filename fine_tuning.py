import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from tqdm import tqdm


class FineTuningDataGenerator:
    """Generate data for fine-tuning LLMs on real estate content"""

    def __init__(self, text_chunks_path="processed_chunks.pkl", output_dir="fine_tuning_data"):
        self.text_chunks_path = text_chunks_path
        self.output_dir = output_dir
        self.chunks_df = None

        # Load chunks data
        if os.path.exists(text_chunks_path):
            self.chunks_df = pd.read_pickle(text_chunks_path)
            print(f"Loaded {len(self.chunks_df)} text chunks for fine-tuning data generation")

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_qa_pairs(self, num_pairs=100, min_context_length=200):
        """
        Generate question-answer pairs for fine-tuning based on chunks.

        This creates synthetic QA pairs from the text chunks.
        """
        if self.chunks_df is None:
            print("No chunks data available. Please check the path.")
            return

        # Filter chunks by minimum length
        valid_chunks = self.chunks_df[self.chunks_df['text'].str.len() > min_context_length]
        if len(valid_chunks) < num_pairs:
            print(f"Warning: Only {len(valid_chunks)} valid chunks available.")
            num_pairs = len(valid_chunks)

        # Sample random chunks
        selected_chunks = valid_chunks.sample(num_pairs)

        qa_pairs = []

        question_templates = [
            "What does the document say about {topic}?",
            "Can you explain {topic} based on the text?",
            "What information is provided about {topic}?",
            "How does the document describe {topic}?",
            "What are the key points mentioned about {topic}?"
        ]

        # Extract topics from chunks and create QA pairs
        for _, chunk in tqdm(selected_chunks.iterrows(), total=len(selected_chunks),
                             desc="Generating QA pairs"):
            # Extract potential topic words (nouns) using simple heuristics
            # In a real implementation, you might use NLP libraries for better extraction
            text = chunk['text']
            words = text.split()

            # Find potential topic words (longer words, likely to be meaningful)
            potential_topics = [word for word in words
                                if len(word) > 5 and word.isalpha()]

            if not potential_topics:
                # If no good topics found, use generic template
                question = "What information is provided in this text?"
                answer = text
            else:
                # Select a random topic and question template
                topic = np.random.choice(potential_topics)
                question_template = np.random.choice(question_templates)
                question = question_template.format(topic=topic)
                answer = text

            qa_pairs.append({
                'question': question,
                'answer': answer,
                'source': chunk['source_file']
            })

        # Save the generated QA pairs
        output_path = os.path.join(self.output_dir, "qa_pairs.json")
        with open(output_path, 'w') as f:
            json.dump(qa_pairs, f, indent=2)

        print(f"Generated {len(qa_pairs)} QA pairs and saved to {output_path}")
        return qa_pairs

    def generate_fine_tuning_formats(self, qa_pairs=None, test_size=0.2):
        """
        Format QA pairs for different fine-tuning methods.

        Generates formats for:
        1. OpenAI fine-tuning (JSONL)
        2. Hugging Face fine-tuning (CSV)
        """
        if qa_pairs is None:
            # Try to load QA pairs
            qa_path = os.path.join(self.output_dir, "qa_pairs.json")
            if os.path.exists(qa_path):
                with open(qa_path, 'r') as f:
                    qa_pairs = json.load(f)
            else:
                print("No QA pairs provided or found. Please generate QA pairs first.")
                return

        # Split into train and test sets
        train_pairs, test_pairs = train_test_split(qa_pairs, test_size=test_size, random_state=42)

        # 1. OpenAI fine-tuning format (JSONL)
        openai_train = []
        openai_test = []

        for pair in train_pairs:
            openai_train.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specialized in real estate."},
                    {"role": "user", "content": pair['question']},
                    {"role": "assistant", "content": pair['answer']}
                ]
            })

        for pair in test_pairs:
            openai_test.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specialized in real estate."},
                    {"role": "user", "content": pair['question']},
                    {"role": "assistant", "content": pair['answer']}
                ]
            })

        # Save OpenAI format
        with open(os.path.join(self.output_dir, "openai_train.jsonl"), 'w') as f:
            for item in openai_train:
                f.write(json.dumps(item) + '\n')

        with open(os.path.join(self.output_dir, "openai_test.jsonl"), 'w') as f:
            for item in openai_test:
                f.write(json.dumps(item) + '\n')

        # 2. Hugging Face fine-tuning format (CSV)
        hf_train_data = pd.DataFrame({
            'input': [pair['question'] for pair in train_pairs],
            'output': [pair['answer'] for pair in train_pairs]
        })

        hf_test_data = pd.DataFrame({
            'input': [pair['question'] for pair in test_pairs],
            'output': [pair['answer'] for pair in test_pairs]
        })

        # Save Hugging Face format
        hf_train_data.to_csv(os.path.join(self.output_dir, "hf_train.csv"), index=False)
        hf_test_data.to_csv(os.path.join(self.output_dir, "hf_test.csv"), index=False)

        print(f"Created fine-tuning datasets in multiple formats:")
        print(f"OpenAI: {len(openai_train)} training, {len(openai_test)} testing examples")
        print(f"Hugging Face: {len(hf_train_data)} training, {len(hf_test_data)} testing examples")

        return {
            'openai': {'train': openai_train, 'test': openai_test},
            'huggingface': {'train': hf_train_data, 'test': hf_test_data}
        }