import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from sklearn.decomposition import PCA, KernelPCA  # For X initialization
from PIL import Image
import PIL.ImageOps
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import pandas as pd
from simple_gplvm import simple_gplvm


def load_genes_dataset():
    URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    df = pd.read_csv(URL, index_col=0)
    N = df.values.shape[0]
    D = df.values.shape[1]
    n_classes = len(df.index.unique())
    return N, n_classes, D, np.asarray(df.values), df.index

def plot_genes(pca, obs, labels=None):
    colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]
    labs = ["1", "2", "4", "8", "16", "32 TE", "32 ICM", "64 PE", "64 TE", "64 EPI"]
    plt.figure(figsize=(7, 5))
    for i, label in enumerate(labs):
        pca_i = pca[labels == label]
        plt.scatter(pca_i[:, 0], pca_i[:, 1], c=colors[i], label=label)

    plt.legend()
    plt.title("PCA Single-Cell qPCR data")
    plt.show()

def plot(pca, gp_vals, labels=None):
    ''' Simple scatter of 2d data in same figure
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(8,6))
    if labels is not None:
        colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]
        for i, label in enumerate(np.unique(labels)):
            pca_i = pca[labels == label]
            gp_i = gp_vals[labels == label]
            ax[0].scatter(pca_i[:, 0], pca_i[:, 1], c=colors[i], label=label)
            ax[1].scatter(gp_i[:, 0], gp_i[:, 1], c=colors[i], label=label)
    else:
        ax[0].scatter(pca[:, 0], pca[:, 1])
        ax[1].scatter(gp_vals[:, 0], gp_vals[:, 1])
    ax[0].grid()
    ax[0].set_title("PCA Single-Cell qPCR data")
    ax[1].grid()
    ax[1].set_title("GPLVM Single-Cell qPCR data")
    fig.legend(np.unique(labels))
    fig.savefig("comparison_pca_gp.png")
    plt.show()

if __name__ == "__main__":

    N, n_classes, D, observations, labels= load_genes_dataset()
    pca = PCA(n_components=2).fit_transform(observations)
    gp_simple = simple_gplvm(Y=observations)
    # print(pca.shape)
    # plot_genes(pca, observations[a,0:10], labels[a])
    # plot_genes(gp_simple, observations[a,0:10], labels[a])
    plot(pca, gp_simple, labels)
