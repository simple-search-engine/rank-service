from sklearn.feature_extraction.text import TfidfVectorizer
from util.request_tokenizer import requestToTokenizer
import numpy as np

tfidf_vectorizer = TfidfVectorizer()

def get_tf_idf_embedding(query, documents): 
    tokenized_corpus = requestToTokenizer(get_corpus(query, documents))
    
    query_vec, doc_vec = get_query_documents_tfidf(tokenized_corpus)
    return query_vec, np.array(doc_vec)

def toList(array) :
    return array.tolist()[0][0]

def get_corpus(query, documents):
    return [query] + documents

def get_tfidf_matrix(tokenized_corpus):
    return tfidf_vectorizer.fit_transform(tokenized_corpus).todense()

def get_query_tfidf(tfidf_matrix):
    return np.asarray(tfidf_matrix[0])

def get_documents_tfidf(tfidf_matrix):
    return [np.asarray(element) for element in tfidf_matrix[1:]]

def get_query_documents_tfidf(tokenized_corpus):
    tfidf_matrix = get_tfidf_matrix(tokenized_corpus)
    query_tfidf = get_query_tfidf(tfidf_matrix)
    documents_tfidf = get_documents_tfidf(tfidf_matrix)

    return query_tfidf, documents_tfidf

def get_similarityArr(query_tfidf, documents_tfidf, get_similarity):
    similarity_arr = []

    for document_tfidf in documents_tfidf:
        similarity_arr.append(toList(get_similarity(query_tfidf, document_tfidf)))

    return similarity_arr

def rank_index(similarity_arr, sorted_similarity_arr): 
    sort_index = []
    for similarity in similarity_arr:
        j = 0
        for sorted_similarity in sorted_similarity_arr:
            if similarity == sorted_similarity:
                sort_index.append(j)
                break
            j = j + 1
    return sort_index

def log_similarity(similarity_type, sort_index, similarity_arr):
    i = 0
    count = len(sort_index)
    for i in range (0, count):
        print(sort_index[i]+1, f"번째 문서 {similarity_type} 유사도: ", similarity_arr[i])
    print()

def get_sort_index(similarity_type, get_similarity, query, documents): 
    tokenized_corpus = requestToTokenizer(get_corpus(query, documents))
    
    query_tfidf, documents_tfidf = get_query_documents_tfidf(tokenized_corpus)

    similarity_arr = get_similarityArr(query_tfidf, documents_tfidf, get_similarity)

    sorted_similarity_arr = sorted(similarity_arr)
    if similarity_type == '코사인':
        sorted_similarity_arr.reverse()

    sort_index = rank_index(similarity_arr, sorted_similarity_arr)
    log_similarity(similarity_type, sort_index, similarity_arr)
    print("sort index: ", sort_index, "\n")
    return sort_index