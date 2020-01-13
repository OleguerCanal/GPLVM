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

def load_genes_dataset(N, D):
    # URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    # df = pd.read_csv(URL, index_col=0)
    df = pd.read_csv("data/mice.csv", index_col=0)
    df = df.sample(n = N, axis = 0, random_state = 1)
    df = df.sample(n = D, axis=1, random_state = 1)
    print(df.head())
    N = df.values.shape[0]
    D = df.values.shape[1]
    n_classes = len(df.index.unique())
    return N, n_classes, D, np.asarray(df.values), df.index

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
    fig.savefig("figures/mice/mice_comparison_pca_gp.png")
    plt.show()

if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(10, 10)
    print("N:", N)
    print("n_classes:", n_classes)
    print("D:", D)
    pca = PCA(n_components=2).fit_transform(observations)
    gp_simple = simple_gplvm(Y=observations, experiment_name="mice")
    print(pca.shape)
    plot(pca, gp_simple, labels)