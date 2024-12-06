from flask import Flask, request
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import numpy as np
from config import Config
from util.embedding_tf_idf import get_sort_index, get_tf_idf_embedding
from util.request_tokenizer import get_query_documents
from util.sort import get_response_sort_index, sort, get_similarity
from util.embedding_bert import get_embedding, get_bert_embedding
import json

app = Flask(__name__)

@app.route('/rank/bert', methods=['POST'])
def rankBert():
    query, documents = get_query_documents(request)
    
    # 문장과 문서 벡터화
    query_embedding = get_embedding(query)
    document_embeddings = np.array([get_embedding(doc) for doc in documents])

    # 코사인 유사도 전 벡터 차원 일치
    query_embedding_2d = query_embedding.flatten()
    document_embeddings_2d = document_embeddings.reshape(document_embeddings.shape[0], -1)

    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding_2d], document_embeddings_2d).flatten()

    # 정렬된 결과와 해당 문서의 원래 인덱스를 저장
    sorted_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # rank 인덱스 계산
    rank_indices = []

    # 결과 출력
    for rank, (original_index, similarity) in enumerate(sorted_results, start=1):
        document = documents[original_index]
        print(f"Rank: {rank}, Similarity: {similarity:.4f}, Document: {document}, Original Index: {original_index}")
        
        rank_indices.append(rank)

    return get_response_sort_index(rank_indices)

@app.route('/rank/cos', methods=['POST'])
def rankCos():
    query, documents = get_query_documents(request)
    sortIndex = get_sort_index('코사인', cosine_similarity, query, documents)

    return get_response_sort_index(sortIndex)

@app.route('/rank/man', methods=['POST'])
def rankMan():
    query, documents = get_query_documents(request)
    sortIndex = get_sort_index('맨하탄', manhattan_distances, query, documents)

    return get_response_sort_index(sortIndex)


@app.route('/rank/ucl', methods=['POST'])
def rankUcl():
    query, documents = get_query_documents(request)
    sortIndex = get_sort_index('유클리드', euclidean_distances, query, documents)

    return get_response_sort_index(sortIndex)

def getJsonSortedResult(sorted_results):
    json_results = [{"document": document, "similarity": float(similarity)} for document, similarity in sorted_results]
    return json.dumps(json_results, indent=2)

@app.route('/rank/bert/v2', methods=['POST'])
def rankBert2():
    query, documents = get_query_documents(request)
    
    # 문장과 문서 벡터화
    (query_embedding, document_embeddings) = get_bert_embedding(query, documents)

    # 유사도 계산
    similarities = get_similarity(cosine_similarity, query_embedding, document_embeddings)

    # 유사도 바탕으로 문서 정렬
    sorted_results = sort(documents, similarities, 'asc')

    return getJsonSortedResult(sorted_results)

@app.route('/rank/cos/v2', methods=['POST'])
def rankCos2():
    query, documents = get_query_documents(request)
    
    # 문장과 문서 벡터화
    (query_embedding, document_embeddings) = get_tf_idf_embedding(query, documents)

    # 유사도 계산
    similarities = get_similarity(cosine_similarity, query_embedding, document_embeddings)

    # 유사도 바탕으로 문서 정렬
    sorted_results = sort(documents, similarities, 'asc')

    return getJsonSortedResult(sorted_results)

@app.route('/rank/man/v2', methods=['POST'])
def rankMan2():
    query, documents = get_query_documents(request)
    
    # 문장과 문서 벡터화
    (query_embedding, document_embeddings) = get_tf_idf_embedding(query, documents)

    # 유사도 계산
    similarities = get_similarity(manhattan_distances, query_embedding, document_embeddings)

    # 유사도 바탕으로 문서 정렬
    sorted_results = sort(documents, similarities, 'desc')
    
    return getJsonSortedResult(sorted_results)


@app.route('/rank/ucl/v2', methods=['POST'])
def rankUcl2():
    query, documents = get_query_documents(request)

    # 문장과 문서 벡터화
    (query_embedding, document_embeddings) = get_tf_idf_embedding(query, documents)

    # 유사도 계산
    similarities = get_similarity(euclidean_distances, query_embedding, document_embeddings)

    # 유사도 바탕으로 문서 정렬
    sorted_results = sort(documents, similarities, 'desc')

    return getJsonSortedResult(sorted_results)

if __name__ == '__main__':
    app.run(host=Config.HOST_IP, port=5001, debug=True) 