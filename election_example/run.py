import os
from networkx.readwrite import json_graph
import networkx as nx
import json 
import matplotlib.pyplot as plt
from election_model import ElectionModel
import argparse


from llms.baichuan import BaichuanLLM
#from llms.ollama_lib import OllamaLLM

from casevo import TotLog

# Get Key From: https://platform.baichuan-ai.com/
API_KEY = 'BAICHUAN_API_KEY'
llm = BaichuanLLM(API_KEY, 5)

#Ollama Interface
#llm = OllamaLLM(5)

parser = argparse.ArgumentParser(description='run the Mesa model.')

# Get Config File
parser.add_argument('filename', metavar='config_file', type=str,
                   help='The config file for the sim')
parser.add_argument('round', metavar='round_num', type=int,
                   help='The round number of the sim')


args = parser.parse_args()

tar_file = args.filename

with open(tar_file, 'r') as f:
    config_file = json.load(f) 

# Get Config of the Network and the Agent Profile
G = json_graph.node_link_graph(config_file['graph'])
person_list = config_file['person']

# Draw the network
nx.draw(G, with_labels=True)
plt.savefig('graph.png')

log_path = './log/'
memory_path = './memory/'


if not os.path.exists(log_path):
    os.mkdir(log_path)
if not os.path.exists(memory_path):
    os.mkdir(memory_path)


TotLog.init_log(len(person_list), if_event=True)





# Init Model
model = ElectionModel(G, person_list, llm)

# Run the Sim
for i in range(args.round):
    model.step()

# Write Log
TotLog.write_log(log_path)