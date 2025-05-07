import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd
import graphviz
import networkx as nx

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] 
    return graph

#############################

data_mpg = pd.read_csv("dataset_original_270411_ingles_sc1.csv", low_memory=False, sep=";")
data_mpg.dropna(inplace=True)
#data_mpg.drop(["COD_SEXO"], axis=1, inplace=True)
print(data_mpg.shape)
data_mpg.head()

#############################

from causallearn.search.ConstraintBased.PC import pc

labels = [f'{col}' for i, col in enumerate(data_mpg.columns)]
data = data_mpg.to_numpy()

cg = pc(data)

from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)


size = "10,10!"  
dpi = 300  
pyd.set_graph_defaults(size=size, dpi=str(dpi), rankdir="TB")


dot_filename = "PC_output_graph.dot"
pyd.write_raw(dot_filename)
print(f"Archivo DOT generado: {dot_filename}")


png_filename = "PC_output_graph.png"
pyd.write_png(png_filename)
print(f"Archivo PNG generado: {png_filename}")


tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.figure(figsize=(120, 120))  
plt.axis('off')
plt.imshow(img)
plt.show()
