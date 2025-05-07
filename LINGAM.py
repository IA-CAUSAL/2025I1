import pandas as pd
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
import matplotlib.pyplot as plt


data = pd.read_csv('dataset_original_270411_ingles_sc1.csv', sep=';')


data_array = data.values


model = lingam.DirectLiNGAM()
model.fit(data_array)


adj_matrix = model.adjacency_matrix_


labels = list(data.columns)


dot = make_dot(model.adjacency_matrix_, labels=labels)


dot_file_path = 'lingam_causal_graph.dot'
dot.save(dot_file_path)


dot.attr(dpi="1200") 
dot.attr(size="600,600!")  
dot.format = 'png'
dot.render('lingam_causal_graph', cleanup=False)  


img = plt.imread('lingam_causal_graph.png')
plt.figure(figsize=(120, 120))  
plt.imshow(img)
plt.axis('off')
plt.show()
