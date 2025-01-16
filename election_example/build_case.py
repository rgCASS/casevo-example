import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import json

#生成图
#节点个数
node_num = 5
graph = nx.complete_graph(node_num)
graph_data = json_graph.node_link_data(graph)

#配置人设
with open('person.json') as f:
    person_data = json.load(f)

output_item = {
    "graph": graph_data,
    "person": person_data[:node_num]
}

#输出实验配置文件
with open('case_lite.json', 'w') as f:
    json.dump(output_item, f, ensure_ascii=False)