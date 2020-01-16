from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from algorithms.fast_gplvm import GPLVM

dirlist = sorted(os.listdir("yalefaces/"))
subjects_dict = defaultdict(list)

data = []
num_faces = 6
for i, file in enumerate(dirlist[2:]):
	sample = file.split('.')
	sample_img = np.asarray(Image.open("yalefaces/" + file).resize((90,60)))
	subjects_dict[sample[1]].append(i)
	data.append(sample_img.reshape(-1))
data = np.array(data)
data = (data)
print(subjects_dict.keys())
#pca_faces = zscore(PCA(n_components=3).fit_transform(data[:11*num_faces]))
#pca_faces = zscore(KernelPCA(n_components=3, kernel='rbf', gamma=1).fit_transform(data[:11*num_faces]))
pca_faces = zscore(GPLVM(latent_dim=3, active_set_size=20).fit_transform(data[:11*num_faces], iterations=200))
print(pca_faces)

labels = np.divide(list(range(num_faces*11)), 11).astype(int)

dic_data = {'data': pca_faces, 'labels': labels}
np.save("results_gplvm.npy", dic_data)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_faces):
	ax.scatter(pca_faces[i*11:(i+1)*11,0], pca_faces[i*11:(i+1)*11,1], pca_faces[i*11:(i+1)*11,2], alpha=0.5, label= str(i))
plt.legend()

plt.xlabel('pc1')
plt.ylabel('pc2')

plt.show()
