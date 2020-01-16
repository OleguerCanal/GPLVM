import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA  # For X initialization
from PIL import Image
import PIL.ImageOps
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import time
from simple_gplvm import simple_gplvm
from fast_gplvm import GPLVM

np.random.seed(1)

def load_digit_dataset(samples):
    path = "data/digits.h5"
    import h5py
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
    D = len(X_tr[0])
    n_classes = 10
    N = len(X_tr)

    Y = np.asarray(X_tr)
    idx = np.random.randint(Y.shape[0], size=samples)

    return Y.shape[0], n_classes, D, Y[idx, :], np.asarray(y_tr)[idx].reshape(-1,1)

def plot_digits(pca, dig_obs, labels=None, name = ""):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    colors = ["blue","red","green","black","yellow","pink","purple","orange","brown", "grey"]

    if labels is not None:
        ax.scatter(pca[:, 0], pca[:, 1], s=np.zeros(len(pca[:,0]))*0.001)
        i = 0
        for x0, y0 in zip(pca[:,0], pca[:,1]):
            if i%5==0:
                im = Image.fromarray(dig_obs[i].reshape(16,16)*255)
                im = im.convert("L")
                im = PIL.ImageOps.invert(im)
                im = PIL.ImageOps.colorize(im, black =colors[labels[i][0]], white ="white") 
                im = im.convert("RGBA")
                datas = im.getdata()
                newData = []
                for item in datas:
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)

                im.putdata(newData)
                # im.save("img2.png", "PNG")
                # inverted_im.save("img.jpg")
                bb = Bbox.from_bounds(x0,y0,0.25,0.25)  
                bb2 = TransformedBbox(bb,ax.transData)
                bbox_image = BboxImage(bb2,
                                    norm = None,
                                    origin=None,
                                    clip_on=False)
                bbox_image.set_data(im)
                ax.add_artist(bbox_image)

            i += 1
    else:
        ax[0].scatter(pca[:, 0], pca[:, 1])
        ax[1].scatter(gp_vals[:, 0], gp_vals[:, 1])
    ax.grid()
    plt.savefig("figures/digits/" + name + "_result_" + str(time.time()) + ".png")
    plt.show()

if __name__ == "__main__":
    N, n_classes, D, dig_obs, dig_lab = load_digit_dataset(samples=300)

    # PCA
    pca_dig = PCA(n_components=2).fit_transform(dig_obs)
    plot_digits(pca_dig, dig_obs, dig_lab, name = "pca")

    # GPLVM experiment:
    for active_set_size in [100, 200]:
        name = "RationalQuadratic_kernel_digits_size_" + str(active_set_size) + "_exp_300_samples_"
        gplvm = GPLVM(active_set_size=active_set_size, name=name)
        gp_vals = gplvm.fit_transform(dig_obs, iterations=100, save = True)
        plot_digits(gp_vals, dig_obs, dig_lab, name=name)

    # Load saved experiment example:
    # gplvm = GPLVM(active_set_size=30, name="digits_exp")
    # gplvm.load("results/digits_exp_2020-01-13_23:17:32.735412")
    # gp_vals = gplvm.X
    # plot_digits(gp_vals, dig_obs, dig_lab, name = "gplvm")


