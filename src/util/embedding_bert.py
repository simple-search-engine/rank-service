from transformers import BertTokenizerFast, BertModel
import torch
import numpy as np

model_name = "kykim/bert-kor-base"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.detach().numpy()

def get_bert_embedding(query, documents):
    query_embedding = get_embedding(query)
    document_embeddings = np.array([get_embedding(doc) for doc in documents])

    return query_embedding, document_embeddings