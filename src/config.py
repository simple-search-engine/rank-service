import os 

PY_ENV = os.environ.get('PY_ENV', 'local')
TOKENIZER_IP = "tokenizer" if (PY_ENV == 'production') else "127.0.0.1"

class Config:
    HOST_IP = "0.0.0.0" if (PY_ENV == 'production') else "127.0.0.1"
    URL_TOKENIZER_MULTI_DOCUMENTS = f"http://{TOKENIZER_IP}:7000/multiDocuments"
    POST_HEADER = {'Content-Type': 'application/json; charset=utf-8'}