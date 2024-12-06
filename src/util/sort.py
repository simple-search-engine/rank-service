import json
from flask import Response
import numpy as np

def get_response_sort_index(sortIndex):
    json_data = json.dumps({"documents_index" : sortIndex})
    return Response(json_data, content_type='application/json')

def get_similarity(f_similarity_criteria, query_embedding, document_embeddings):
    # 벡터 차원 일치
    query_embedding_2d = query_embedding.flatten()
    array_size = document_embeddings.size
    if array_size > 0:
        document_embeddings_2d = document_embeddings.reshape(document_embeddings.shape[0], -1)
    else:
        return np.array([])
    
    return f_similarity_criteria([query_embedding_2d], document_embeddings_2d).flatten()

def sort(documents, similarities, direction):
    reverse = True if direction == 'asc' else False
    return sorted(zip(documents, similarities), key=lambda x: x[1], reverse=reverse)