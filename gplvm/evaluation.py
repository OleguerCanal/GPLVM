import numpy as np
from sklearn.cluster import KMeans
from fast_gplvm import GPLVM
from sklearn.decomposition import PCA  # For X initialization
from matplotlib import pyplot as plt
from mice_genes import load_genes_dataset
from sklearn.metrics import confusion_matrix

class Evaluation:

    def __init__(self, observations, labels):
        self.observations = observations
        self.labels = labels
        self.N = observations.shape[0]
        self.unique_labels = np.unique(labels)
        self.n_classes = len(unique_labels)
        self.gp_vals = None
        self.pca_vals = None

    def load_gplvm(self, filename):
        self.gp_vals = GPLVM().load(filename)

    def compute(self, active_set_size):
        if self.gp_vals is not None:
            self.gp_vals = GPLVM(active_set_size=active_set_size).fit_transform(observations)
        pca_vals = PCA(n_components=2).fit_transform(observations)

    def cluster(self, points, algorithm='kmeans'):
        # Valid algorithms are 'kmeans' and 'gaussmix'
        # Compute K-means for GPLVM
        if algorithm == 'kmeans':
            gp_kmeans = KMeans(n_clusters=self.n_classes, random_state=0).fit(self.gp_vals)
        else:
            raise Exception('Not implemented yet')

        gp_assignments = gp_kmeans.predict(self.gp_vals)
        gp_clusters = self.__label_clusters(gp_assignments, self.labels)
        self.gp_assignments = np.array([gp_clusters[i] for i in gp_assignments]).astype(int)

        # Compute K-means for PCA and assign lables to clusters correctly
        pca_kmeans = KMeans(n_clusters=self.n_classes, random_state=0).fit(self.pca_vals)
        pca_assignments = pca_kmeans.predict(self.pca_vals) 
        pca_clusters = self.__label_clusters(pca_assignments, self.labels)
        self.pca_assignments = np.array([pca_clusters[i] for i in pca_assignments]).astype(int)

    def __label_clusters(self, assigned_labels, true_lables):
        # Assign to each lable the cluster that contains the highest share of its points
        cluster_assignments = np.empty(self.n_classes)
        for k in range(self.n_classes):
            total = np.sum(np.where(assigned_labels == k, 1, 0))
            scores = np.zeros(self.n_classes)
            for i, label in enumerate(self.unique_labels):
                scores[i] = np.sum(np.where((k_means_labels == k) * (true_lables == label), 1, 0))
            scores /= total
            cluster_assignments[k] = np.argmax(scores)
        return cluster_assignments

    def get_confusion_matrices(self):
        gp_confusion_matrix = confusion_matrix(labels, [self.unique_labels[i]
                                                        for i in self.gp_assignments])
        pca_confusion_matrix = confusion_matrix(labels, [self.unique_labels[i]
                                                         for i in self.pca_assignments])
        return gp_confusion_matrix, pca_confusion_matrix
    
    def plot(self):
        colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]
        fig, axes = plt.subplots(2,2)

        # Plot pca and gplvm
        for i, label in enumerate(self.unique_labels):
            pca_points = self.pca_vals[self.labels == label]
            axes[0, 0].scatter(pca_points[:,0], pca_points[:,1], c=colors[i])
            axes[0, 0].set_title('PCA')

            gplvm_points = self.gp_vals[self.labels == label]
            axes[1, 0].scatter(gplvm_points[:, 0], gplvm_points[:, 1], c=colors[i])
            axes[1, 0].set_title('GPLVM')

        for k in self.unique_labels:
            pca_k_points = self.pca_vals[self.pca_assignments == k]
            axes[0, 1].scatter(pca_k_points[:,0], pca_k_points[:,1], c=colors[k])
            axes[0, 1].set_title('K means reconstruction')

            gplvm_k_points = self.gp_vals[self.gp_assignments == k]
            axes[1, 1].scatter(gplvm_k_points[:,0], gplvm_k_points[:,1], c=colors[k])
            axes[1, 1].set_title('K means reconstruction') 
        plt.show()

    def plot_confusion_matrix():
        fig, axes = plt.subplots(1,2)
        confusion_matrices = self.get_confusion_matrices()
        for idx, cm in enumerate(confusion_matrices):
            axes[0,idx].imshow(cm,interpolation='none',cmap='Blues')
            for (i, j), z in np.ndenumerate(cm):
                axes[0, idx].text(j, i, z, ha='center', va='center')
            axes[0, idx].xlabel("kmeans label")
            axes[0, idx].ylabel("truth label")
            axes[0, idx].title(title)
        plt.show()
 
 if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(100,15)
    observations = observations - np.mean(observations, axis=0)
    
    plot(labels, gp_vals, gp_assignments, pca_vals, pca_assignments)

    plot_confusion_matrix(gp_confusion_matrix, "GPLVM Confusion Matrix")
    plot_confusion_matrix(pca_confusion_matrix, "PCA Confusion Matrix")
