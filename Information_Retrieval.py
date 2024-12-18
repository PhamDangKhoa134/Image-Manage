from transformers import AutoTokenizer, AutoModel
import torch
import re

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert = AutoModel.from_pretrained("vinai/phobert-base")

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def encode_sentences(sentences, tokenizer, model):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

file_path = 'pred_captions.txt'
content = read_file(file_path)
sentences = re.split(r'\.\s*|\n', content)
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

# query = "một người đàn ông"
query = input("Nhập từ cần tìm kiếm: ")
encoded_query = encode_sentences([query], tokenizer, phobert)
encoded_sentences = encode_sentences(sentences, tokenizer, phobert)

similarities = cosine_similarity(encoded_query, encoded_sentences)
sorted_sentences = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

found = False
for idx, score in sorted_sentences[:10]:
    if score > 0:
        print(f"Suitable sentence {idx + 1}: {sentences[idx]}")
        print(f"Cosine Similarity Score: {score.item()}")
        print()
        found = True

if not found:
    print("Không có tìm kiếm phù hợp")
