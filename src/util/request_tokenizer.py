import requests
from config import Config

def requestToTokenizer(documents):
    datas = { "documents" : documents }

    response = requests.post(url=Config.URL_TOKENIZER_MULTI_DOCUMENTS, headers=Config.POST_HEADER, json=datas)
    return response.json()['tokenizedDocuments']

def get_query_documents(request):
    payload = request.json
    query = payload['query']
    documents = payload['documents']
    print("\n" + "query     : " + query)
    print("documents :", documents, '\n')
    return query, documents