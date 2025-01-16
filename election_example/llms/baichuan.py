from chromadb import Documents, EmbeddingFunction, Embeddings
from casevo import LLM_INTERFACE
import requests
import time
import json



URL = 'https://api.baichuan-ai.com/v1/chat/completions'

EMBEDDING_URL = 'http://api.baichuan-ai.com/v1/embeddings'


class BaichuanEmbedding(EmbeddingFunction):
    def __init__(self,llm, tar_len):
        #super.__init__()
        self.SEND_LEN = tar_len
        self.llm = llm
    
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        res_list = []
        cur_list = []
        for item in input:
            cur_list.append(item)
            if len(cur_list) >= self.SEND_LEN:
                res = self.llm.send_embedding(cur_list)
                res_list.extend(res)
                cur_list = []
        if len(cur_list) > 0:
            res = self.llm.send_embedding(cur_list)
            res_list.extend(res)

        return res_list


class BaichuanLLM(LLM_INTERFACE):
    def __init__(self, tar_key, tar_len):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + tar_key
        }
        self.embedding_len = tar_len
        self.embedding_function = BaichuanEmbedding(self, self.embedding_len)
    
    def send_message(self, prompt, json_flag=False):
        data = {
            'model': 'Baichuan4-Air',
            'messages': [{'role':'user','content':prompt}],
            'temperature': 0.3,
            'top_p': 0.85,
            'max_tokens': 2048,
            'with_search_enhance': False
        }
        response = requests.post(URL, headers=self.headers, json=data)
        while response.status_code == 429:
            print('API rate limit exceeded, waiting 10 seconds...')
            time.sleep(5)
            response = requests.post(URL, headers=self.headers, json=data)
        if response.status_code == 200:
            result = json.loads(response.text)
            if result['choices'][0]['finish_reason'] == 'content_filter':
                print('Content filtered, retrying...')
                time.sleep(5)
                response = requests.post(URL, headers=self.headers, json=data)
                result = json.loads(response.text)
            tmp_content = result['choices'][0]['message']['content'].strip()
            #time.sleep(5)
            return tmp_content
        else:
            return None

    def send_embedding(self, text_list):
        data = {
            'model': 'Baichuan-Text-Embedding',
            'input': text_list
        }
        response = requests.post(EMBEDDING_URL, headers=self.headers, json=data)
        #print(response)
        while response.status_code == 429:
            print('API rate limit exceeded, waiting 10 seconds...')
            time.sleep(5)
            response = requests.post(EMBEDDING_URL, headers=self.headers, json=data)
        if response.status_code == 200:
            result = json.loads(response.text)
            return [item['embedding'] for item in result['data']]
        else:
            return None
    
    def get_lang_embedding(self):
        
        return self.embedding_function