import numpy as np
from sklearn.cluster import KMeans
from fast_gplvm import GPLVM
from sklearn.decomposition import PCA  # For X initialization
from matplotlib import pyplot as plt
from mice_genes import load_genes_dataset

def plot(labels, gp_vals, gp_assignments, pca_vals, pca_assignments):
    colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]
    fig, axes = plt.subplots(2,2)

    # Plot pca and gplvm
    for i, label in enumerate(np.unique(labels)):
        pca_points = pca_vals[labels == label]
        axes[0, 0].scatter(pca_points[:,0], pca_points[:,1], c=colors[i])

        gplvm_points = gp_vals[labels == label]
        axes[1, 0].scatter(gplvm_points[:, 0], gplvm_points[:, 1], c=colors[i])

    for k in np.unique(pca_assignments):
        k = int(k)
        pca_k_points = pca_vals[pca_assignments == k]
        axes[0, 1].scatter(pca_k_points[:,0], pca_k_points[:,1], c=colors[k])

        gplvm_k_points = gp_vals[gp_assignments == k]
        axes[1, 1].scatter(gplvm_k_points[:,0], gplvm_k_points[:,1], c=colors[k])

    
    plt.show()

def assign_clusters_to_lables(k_means_labels, true_lables):
    # Assign to each lable the cluster that contains the highest share of its points
    n_clusters = np.unique(k_means_labels).shape[0]
    cluster_assignments = np.empty(n_clusters)
    for i, label in enumerate(np.unique(true_lables)):
        scores = np.empty(n_clusters)
        for k in range(n_clusters):
            # Indexes of points assigned to cluster k
            idx = np.where(k_means_labels == k)[0]
            correct = len(np.where(labels[idx] == label))
            total = len(np.where(k_means_labels == k)[0])
            scores[k] = correct / total 
        
        # Label i is assigned to cluster with higher score
        cluster_assignments[i] = int(np.argmax(scores))
    return cluster_assignments

def confusion_matrix():
    pass
        
if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(100,15)
    observations = observations - np.mean(observations, axis=0)

    # Compute GPLVM latent space
    gp_vals = GPLVM(active_set_size=int(N * 0.6)).fit_transform(observations, iterations=10)

    # Compute PCA latent space
    pca_vals = PCA(n_components=2).fit_transform(observations)

    # Compute K-means for GPLVM
    gp_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(gp_vals)
    gp_assignments = gp_kmeans.predict(gp_vals)

    # Compute K-means for PCA
    pca_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(pca_vals)
    pca_assignments = pca_kmeans.predict(pca_vals) 

    cluster_assignments = assign_clusters_to_lables(pca_assignments, labels)
    #pca_assignments = np.array([cluster_assignments[i] for i in pca_assignments])
    
    plot(labels, gp_vals, gp_assignments, pca_vals, pca_assignments)
