import numpy as np
from sklearn.cluster import KMeans
from fast_gplvm import GPLVM
from sklearn.decomposition import PCA  # For X initialization
from matplotlib import pyplot as plt
from mice_genes import load_genes_dataset
from sklearn.metrics import confusion_matrix

def plot(labels, gp_vals, gp_assignments, pca_vals, pca_assignments):
    colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]
    fig, axes = plt.subplots(2,2)

    # Plot pca and gplvm
    for i, label in enumerate(np.unique(labels)):
        pca_points = pca_vals[labels == label]
        axes[0, 0].scatter(pca_points[:,0], pca_points[:,1], c=colors[i])
        axes[0, 0].set_title('PCA')

        gplvm_points = gp_vals[labels == label]
        axes[1, 0].scatter(gplvm_points[:, 0], gplvm_points[:, 1], c=colors[i])
        axes[1, 0].set_title('GPLVM')

    for k in np.unique(pca_assignments):
        k = int(k)
        pca_k_points = pca_vals[pca_assignments == k]
        axes[0, 1].scatter(pca_k_points[:,0], pca_k_points[:,1], c=colors[k])
        axes[0, 1].set_title('K means reconstruction')

        gplvm_k_points = gp_vals[gp_assignments == k]
        axes[1, 1].scatter(gplvm_k_points[:,0], gplvm_k_points[:,1], c=colors[k])
        axes[1, 1].set_title('K means reconstruction')

    
    plt.show()

def label_clusters(k_means_labels, true_lables):
    # Assign to each lable the cluster that contains the highest share of its points
    n_clusters = np.unique(k_means_labels).shape[0]
    cluster_assignments = np.empty(n_clusters)
    for k in range(n_clusters):
        total = np.sum(np.where(k_means_labels == k, 1, 0))
        scores = np.zeros(n_clusters)
        for i, label in enumerate(np.unique(true_lables)):
            scores[i] = np.sum(np.where((k_means_labels == k) * (true_lables == label), 1, 0))
        scores /= total
        cluster_assignments[k] = np.argmax(scores)
    return cluster_assignments
     

def plot_confusion_matrix(matrix, title):
    # Plot confusion matrix
    plt.imshow(matrix,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(matrix):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("kmeans label")
    plt.ylabel("truth label")
    plt.title(title)
    plt.show()
        
if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(100,15)
    observations = observations - np.mean(observations, axis=0)
    unique_labels = np.unique(labels)

    # Compute GPLVM latent space
    gp_vals = GPLVM(active_set_size=int(N * 0.6)).fit_transform(observations, iterations=10)

    # Compute PCA latent space
    pca_vals = PCA(n_components=2).fit_transform(observations)

    # Compute K-means for GPLVM
    gp_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(gp_vals)
    gp_assignments = gp_kmeans.predict(gp_vals)
    gp_clusters = label_clusters(gp_assignments, labels)
    gp_assignments = np.array([gp_clusters[i] for i in gp_assignments]).astype(int)

    # Compute K-means for PCA and assign lables to clusters correctly
    pca_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(pca_vals)
    pca_assignments = pca_kmeans.predict(pca_vals) 
    pca_clusters = label_clusters(pca_assignments, labels)
    pca_assignments = np.array([pca_clusters[i] for i in pca_assignments]).astype(int)

    gp_confusion_matrix = confusion_matrix(labels, [unique_labels[i] for i in gp_assignments])
    pca_confusion_matrix = confusion_matrix(labels, [unique_labels[i] for i in pca_assignments])
    
    plot(labels, gp_vals, gp_assignments, pca_vals, pca_assignments)

    plot_confusion_matrix(gp_confusion_matrix, "GPLVM Confusion Matrix")
    plot_confusion_matrix(pca_confusion_matrix, "PCA Confusion Matrix")

