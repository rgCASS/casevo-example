from chromadb import Documents, EmbeddingFunction, Embeddings
from casevo import LLM_INTERFACE

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


BASE_URL = ''

# Model Name
CHAT_MODEL = 'qwen2:7b'

EMBEDDING_MODEL= 'znbang/bge:large-zh-v1.5-f16'



class MyEmbedding(EmbeddingFunction):
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


class OllamaLLM(LLM_INTERFACE):
    def __init__(self, tar_len):
        
        self.embedding_len = tar_len
        self.embedding_function = MyEmbedding(self, self.embedding_len)
        self.chat_client = Ollama(model=CHAT_MODEL, 
                                  base_url=BASE_URL, 
                                  temperature=0.3,
                                  request_timeout=60)
        self.embedding_client = OllamaEmbedding(model_name=EMBEDDING_MODEL,
                                                base_url=BASE_URL)
    
    def send_message(self, prompt, json_flag=False):
        #print(prompt)
        resp = None
        for i in range(2):
            try:
                resp = self.chat_client.complete(prompt)
                break
            except Exception as e:
                print(e)
                print('Send Message Retry %d' % i)
                #time.sleep(1)
        #print(resp.text)
        #print('-----------')
        if resp:
            return resp.text
        else:
            raise Exception('send message error')

    def send_embedding(self, text_list):
        
        pass_embedding = None
        for i in range(2):
            try:
                pass_embedding = self.embedding_client.get_text_embedding_batch(text_list)
                break
            except Exception as e:
                print(e)
                print('Send Embedding Retry %d' % i)
                #time.sleep(1)

        if pass_embedding:
            return pass_embedding
        else:
            raise Exception('Send embedding error')
        #print(response)
        
    def get_lang_embedding(self):
        
        return self.embedding_function