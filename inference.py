from transformers import pipeline
import torch
import sys
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA as sklearnPCA
sys.path.append("/home/nasim/project/affect-similarity/py_isear_dataset")
from py_isear.isear_loader import IsearLoader
import pandas as pd
import random
from collections import Counter

#setting uop the pipelines for inference from the two emotion classification models
device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
pipe_go = pipeline(
    "text-classification", model="go_emotions_8", device=device
)
pipe_dair = pipeline(
    "text-classification", model="dair_emotion_8", device=device
)

attributes = ['INTS']
target = ['EMOT']
loader = IsearLoader(attributes, target, True)
data = loader.load_isear('/home/nasim/project/affect-similarity/py_isear_dataset/isear.csv')

real_labels = data.get_target()
intensity = data.get_data()
texts = data.get_freetext_content()

num_samples = len(texts)
print("Number of samples: ", str(num_samples))

go_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
              'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
              'disapproval', 'disgust', 'embarrassment', 'excitement',
              'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 
              'optimism', 'pride', 'realization', 'relief', 'remorse', 
              'sadness', 'surprise', 'neutral']

dair_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

vectors_both = []
vectors_go = []
vectors_dair = []
vectors_target = []

with open("texts_and_labels.txt", mode="wt") as f:
    header = "text|target_emotion|intensity|"
    for l in go_labels:
        header += "{}|".format(l)
    for l in dair_labels:
        header += "{}|".format(l)
    header += "\n"
    f.write(header)
    
    for iter in tqdm(range(num_samples)):
        texts[iter] = texts[iter].replace('á ', '')
        res  = "{}|{}|{}|".format(texts[iter], real_labels[iter][0], intensity[iter][0])
        
        go_res = pipe_go(texts[iter], return_all_scores = True)
        go_res = {int(s['label'][6:]):s['score'] for s in go_res[0]}
        go_res_vec = [round(x, 5) for x in go_res.values()]
        for i in range(len(go_labels)):
            res += "{}|".format(round(go_res[i], 5))
            
        dair_res = pipe_dair(texts[iter], return_all_scores = True)
        dair_res = {s['label']:s['score'] for s in dair_res[0]}
        dair_res_vec = [round(x, 5) for x in dair_res.values()]
        for l in dair_labels:
            res += "{}|".format(round(dair_res[l], 5))
            
        res += "\n"
        
        vectors_both.append(go_res_vec + dair_res_vec)
        vectors_go.append(go_res_vec)
        vectors_dair.append(dair_res_vec)
        
        f.write(res)

f.close()


def transform_vectors(vectors):
    """Transforms the mutidimantional vectors (34 dimentions) representing each sample
    to a 2 dimentional vector using PCA (Principal Component Analysis) for visualizing 
    the calculated k_means 

    Args:
        vectors (list[float]): a num_sample x 34|28|6 dimensional 
        matrix of the probability of each classification for each 
        sample with either both models trained on go_emotions and 
        dair_emotions or one of them

    Returns:
        _pandas dataframe_: the most accurate 2 dimentional representation
        of vector representing all the classifications
    """
    vectors = np.array(vectors)
    v_norm = (vectors - vectors.min())/(vectors.max() - vectors.min())
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(v_norm))
    return transformed

t_vectors_both = transform_vectors(vectors_both)
t_vectors_go = transform_vectors(vectors_go)
t_vectors_dair = transform_vectors(vectors_dair)


#finding the number of k_means clusters with the least amount of inertia and min number of classifications 
def find_least_inertia_k_means(vectors, filename, x_line):   
    wcss = [] 
    for i in range(1, 11): 
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
        kmeans.fit(vectors) 
        wcss.append(kmeans.inertia_)
        
    clusters = [i for i in range(1, 11)]
    plt.plot(clusters, wcss)
    plt.axvline(x=x_line, linestyle="dashed", color='red') 
    plt.savefig(filename)
    plt.close()
    
find_least_inertia_k_means(vectors_both, 'cluster_inertia_go_and_dair.png', 7)
find_least_inertia_k_means(vectors_go, 'cluster_inertia_go.png', 7)
find_least_inertia_k_means(vectors_dair, 'cluster_inertia_dair.png', 7)

N_CLUSTERS = 7

def calculate_kmeans_draw_plt(vectors, t_vectors, filename):
    """calculates the k_means for each of the multidimensional classification data represnting
    each sample. And draws the plot with the 2 dimensional representations calculated with pca.

    Args:
        vectors (list[float]): a num_sample x 34|28|6 dimensional 
        matrix of the probability of each classification for each 
        sample with either both models trained on go_emotions and 
        dair_emotions or one of them
        t_vectors (_type_): The 2 dimensional representation of 
        the same vectors calculated with pca
        filename (_type_): where to store the figure

    Returns:
        _list[int]_: a list (with length num_samples) of k_means 
        labels for wach vector of classification probabilities 
        of each sample
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=1) 
    kmeans.fit(vectors)
    k_labels = kmeans.labels_
    sns.scatterplot(x = t_vectors[0], y = t_vectors[1], hue=kmeans.labels_, 
                    palette=['green','orange','brown','dodgerblue','red', "black", "purple"], s = 10)
    plt.savefig(filename+'.png')
    plt.close()
    return k_labels

k_lables_both = calculate_kmeans_draw_plt(vectors_both, t_vectors_both, 'k_means_both')
k_lables_go = calculate_kmeans_draw_plt(vectors_go, t_vectors_go, 'k_means_go')
k_lables_dair = calculate_kmeans_draw_plt(vectors_dair, t_vectors_dair, 'k_means_dair')

def label_histograms(k_labels, filename):
    """Draws the histogram of the relative frequency of target emotions in the ISEAR dataset in the each cluster 
    of the given k-means result.

    Args:
        k_labels (list[int]): the k-means label for each data entry in the ISEAR dataset
        filename (str): where to store the histogram
    """
    seperated_labels = [[] for i in range(N_CLUSTERS)]
    for i in range(len(k_labels)):
        seperated_labels[k_labels[i]].append(i)

    #used instead of counter() to ensure all labels are listed in the histogram
    #even if the count for that lable is 0
    base_dict_counts = {i:0 for i in range(1, 8)}
    counts = [base_dict_counts.copy() for i in range(N_CLUSTERS)]
    for i in range(N_CLUSTERS):
        for j in range(len(seperated_labels[i])):
            index = seperated_labels[i][j]
            real_label_index = real_labels[index][0]
            counts[i][real_label_index] += 1
    
    target_labels = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    
    for i in range(N_CLUSTERS):
        plt.bar(target_labels, counts[i].values())
        plt.title(filename+ str(i))
        plt.xlabel("target emotion labels in ISEAR dataset")
        plt.ylabel("number of samples in cluster {} with each emotion label".format(i))
        plt.savefig("histograms/" + filename+ str(i) + '.png')
        plt.close()
        
label_histograms(k_lables_both, "both_histogram_cluster_")
label_histograms(k_lables_go, "go_histogram_cluster_")   
label_histograms(k_lables_dair, "dair_histogram_cluster_") 


def draw_random_from_clusters(k_labels, file_name):
    """For each label group, draws 10 random pairs from the same label
    and one sample paired with a sample from a different label group.

    Args:
        k_labels (list[int]): list of k-means labels for each sample 
        file_name (str): where to store the the samples
    """
    with open(file_name, mode="wt") as f:
        seperated_labels = [[] for i in range(N_CLUSTERS)]
        for i in range(len(k_labels)):
            seperated_labels[k_labels[i]].append(i)
        
        for i in range(N_CLUSTERS):
            res = "k means label: {} \n".format(i)
            res += "Samples from the same cluster: \n"
            for _ in range(10):
                rand_i = random.randint(0, len(seperated_labels[i])-1)
                rand_j = random.randint(0, len(seperated_labels[i])-1)
                while rand_j == rand_i:
                    rand_j = random.randint(0, len(seperated_labels[i]))
                res += "Score:  "
                res += "1){}|{}|  ".format(texts[seperated_labels[i][rand_i]], i)
                res += "2){}|{}|\n".format(texts[seperated_labels[i][rand_j]], i)
            
            res += "\n"  
            res += "Samples from the different cluster: \n"  
            for j in range(N_CLUSTERS):
                if j == i:
                    continue
                rand_i = random.randint(0, len(seperated_labels[i])-1)
                rand_j = random.randint(0, len(seperated_labels[j])-1)
                res += "Score:  "
                res += "1){}|{}|  ".format(texts[seperated_labels[i][rand_i]], i)
                res += "2){}|{}|\n".format(texts[seperated_labels[j][rand_j]], j)
                
            res += "\n\n\n"
            f.write(res)

    f.close()
            
draw_random_from_clusters(k_lables_both, "samples_both.txt")
draw_random_from_clusters(k_lables_go, "samples_go.txt")   
draw_random_from_clusters(k_lables_dair, "samples_dair.txt")        
            




