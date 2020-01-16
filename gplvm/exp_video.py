from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
from algorithms.fast_gplvm import GPLVM
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from sklearn.decomposition import KernelPCA, PCA

exp_name = "gplvm_r_40"

np.random.seed(3)
size = 200
num_frames = size + 20
data = []
r = 40

y = 30/2
x = -10
for i in range(num_frames):
	img = Image.new('L',size=(size, size))
	draw = ImageDraw.Draw(img)
	leftUpPoint = (x-r, y-r)
	rightDownPoint = (x+r, y+r)
	twoPointList = [leftUpPoint, rightDownPoint]
	draw.ellipse(twoPointList, fill=(1))
	img = img.crop(box=[0,0,size, 30])

	data.append(np.array(img).reshape(-1))
	x += 1
data = np.array(data).astype(float)
index = np.array(list(range(len(data))))

np.random.shuffle(index)
data = data[index]

pca_video = GPLVM(active_set_size=20, name="video_ball", latent_dim=1).fit_transform(data, iterations=25)
#pca_video = KernelPCA(n_components=1, kernel='sigmoid', gamma=15).fit_transform(data)
data = data.astype(np.uint8)
dict_pca = {k:pca_video[k] for k in index}

ordered_index = dict(sorted(dict_pca.items(), key=lambda item: item[1]))

os.makedirs(exp_name + "/result",exist_ok=True)
os.makedirs(exp_name + "/input", exist_ok=True)
for i, k in enumerate(ordered_index.keys()):
	out_file = exp_name + "/result/frame_{:0>6d}.jpg".format(i)
	in_file = exp_name + "/input/frame_{:0>6d}.jpg".format(i)
	img = Image.fromarray(255*data[k].reshape((30, size)))
	img.save(out_file)
	img = Image.fromarray(255*data[i].reshape((30, size)))
	img.save(in_file)

#printing result
os.system('ffmpeg -y -r 30 -i ' + exp_name + '/input/frame_%06d.jpg -c:v h264 -pix_fmt yuv420p -crf 23 ' + exp_name + '/input.mp4')

os.system('ffmpeg -y -r 30 -i ' + exp_name + '/result/frame_%06d.jpg -c:v h264 -pix_fmt yuv420p -crf 23 ' + exp_name + '/output.mp4')
