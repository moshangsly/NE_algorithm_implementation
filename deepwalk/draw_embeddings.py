import matplotlib.pyplot as plt
import numpy as np

embedding_file = "output.embedding"
label_file = "karate.label"

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
            label_list[int(label)-1].append(int(node)-1)

colors = ['m','b','g','c']
labels = [""] * len(graph_list)
for i in range(len(label_list)):
    list = label_list[i]
    for node in list:
        labels[node] = colors[int(i)]
print(labels)

text = np.arange(34)

graph = np.array(graph_list)

x_data, y_data = graph[:, 0], graph[:, 1]

fig,ax = plt.subplots()
ax.scatter(x_data, y_data,c=labels)

plt.show()