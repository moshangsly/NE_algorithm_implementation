import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec

# 设置的参数
adjlist_path = "karate.adjlist"
label_path = "karate.label"
output_path = "output.embedding"
path_num = 10
path_length = 30
window_size = 4
representation_size = 2



class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def random_walk(self, path_length, rand=random.Random(), alpha=0, start=None):
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            curr_node = path[-1]
            if len(G[curr_node]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[curr_node]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def construct_graph(file):
    adjlist = []

    # 将邻接表转换成列表，将邻接表中每一行分割成整数编号的列表
    with open(file) as f:
        for line in f:
            line_list = [int(x) for x in line.strip().split()]
            adjlist.extend([line_list])
    # 将列表转换成图
    G = Graph()
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors
    return G


def sample_random_walks(G, path_num, path_length, alpha=0, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())

    for i in range(path_num):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand, alpha, node))

    return walks


def draw_embeddings(embedding_file, label_file):
    graph_list = []
    label_list = []
    with open(embedding_file) as f:
        first_line = f.readline()
        node_num = first_line.split()[0]
        for i in range(int(node_num)):
            graph_list.append([])
        for line in f:
            node, embeddings = line.split()[0], line.split()[1:]
            for embedding in embeddings:
                graph_list[int(node) - 1].append(float(embedding))

    for i in range(4):
        label_list.append([]),
    with open(label_file) as f:
        for line in f:
            label, nodes = line.split()[0], line.split()[1:]
            for node in nodes:
                label_list[int(label) - 1].append(int(node) - 1)

    colors = ['m', 'b', 'g', 'c']
    labels = [""] * len(graph_list)
    for i in range(len(label_list)):
        list = label_list[i]
        for node in list:
            labels[node] = colors[int(i)]
    graph = np.array(graph_list)
    x_data, y_data = graph[:, 0], graph[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, c=labels)

    plt.show()


# 构建图
G = construct_graph(adjlist_path)

# 采样随机游走
walks = sample_random_walks(G, path_num, path_length)
# 调用word2vec
model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1)
model.wv.save_word2vec_format(output_path)
print("embedding has been saved at ", output_path)
draw_embeddings(output_path,label_path)

