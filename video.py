from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

dirlist = sorted(os.listdir("mcem0_sx408/"))

files = sorted(dirlist)
data = []
for i, file in enumerate(files):
	img = np.asarray(Image.open("mcem0_sx408/" + file).convert('L').crop(box=[210, 120, 310, 265]).resize((20,28)))

	data.append(img.reshape(-1))
data = zscore(np.array(data))
index = np.array(list(range(len(files))))
np.random.shuffle(index)
print(data[0].shape)
data = data[index]

pca_video = KernelPCA(n_components=1, kernel='linear', alpha=0.225, gamma=0.774).fit_transform(data)
dict_pca = {k:pca_video[k] for k in index}

ordered_index = dict(sorted(dict_pca.items(), key=lambda item: item[1]))
print(ordered_index)

os.makedirs("result",exist_ok=True)
for i, k in enumerate(ordered_index.keys()):
	out_file = "result/frame_{:0>6d}.jpg".format(i)
	Image.open("mcem0_sx408/" + files[k]).save(out_file)

#printing result

os.system('ffmpeg -y -r 30 -i result/frame_%06d.jpg -c:v h264 -pix_fmt yuv420p -crf 23 out.mp4')

