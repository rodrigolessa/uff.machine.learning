import numpy as np
import pandas as pd
import os
import shutil
import argparse
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, Birch

def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imgs_folder" , required = True, help = "Images folder")
ap.add_argument("-c", "--clustering_folder" , required = True, help = "Clustering folder")
ap.add_argument("-m", "--moments_folder" , required = True, help = "Moments folder")
ap.add_argument("-n", "--n_clusters", required = True, help = "Number of clusters")
args = vars(ap.parse_args())
imgs_folder = args['imgs_folder']
clustering_folder = args['clustering_folder']
moments_folder = args['moments_folder']
n_clusters = int(args['n_clusters'])

# Zernike orders that will be validated
degrees = [11, 23, 43, 50]

# KMeans params
methods = {
    'KM': KMeans(n_clusters=n_clusters, random_state=0),
    # 'AP': AffinityPropagation()
}

create_folder(clustering_folder)

# get order result for each method
for method in methods: 
    print("Clustering {}".format(method))

    method_folder = create_folder("{}/{}".format(clustering_folder, method))  # create clustering method folder
    
    for degree in degrees:

        print("- order: {}".format(degree))

        # open moments dataset for current degree
        dataset = pd.read_csv(open("{}/moments{}.csv".format(moments_folder, degree)), header=None, sep=';', index_col=None)
        features = dataset.iloc[:, 1:]
        imgs = dataset[dataset.columns[0]].values

        degree_folder = "{}/{}/{}".format(clustering_folder, method, degree)  # create degree folder for current method
        create_folder(degree_folder)

        clustering = methods[method].fit_predict(features)

        # clustering result
        i = 0
        qt = len(imgs)
        for cluster, img in zip(clustering, imgs):

            if i%100 == 0:
                print("{} de {}".format(i, qt))
            i += 1

            dest_folder = "{}/{}".format(degree_folder, cluster)
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)

            img_path = "{}/{}".format(imgs_folder, img)
            
            shutil.copy(img_path, dest_folder)
