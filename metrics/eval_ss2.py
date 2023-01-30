import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
import glob
def list_only_images(dir,extensions):
    images_list=[]
    for extension in extensions:
        images_list.extend(glob.glob(dir+extension))

    images_ids=[x.split("/")[-1] for x in images_list]

    print(images_ids,"\n")
    return images_ids


parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', help='Path to the run folder', type=str)
parser.add_argument('--gt_im_path', help='Path to the mask folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)

#extra?
parser.add_argument('--run_name', help='Path to the gt folder', type=str)
parser.add_argument('--save_path', help='Path to the gt folder', type=str)
parser.add_argument('--gt_label_path', help='txt input directory.')

parsed_args = parser.parse_args()

run_path = parsed_args.pred_path
mask_path = parsed_args.gt_im_path
shape = parsed_args.shape

IMG_WIDTH = shape
IMG_HEIGHT = shape

extensions=["/*.jpg","/*.JPG","/*.png","/*.PNG"]

# path_grey = os.path.join(run_path,"inference/")
path_grey = run_path

print("RUN PATH: ",path_grey)

grey_list = sorted(list_only_images(path_grey,extensions))
img = imread(os.path.join(path_grey, grey_list[0]))


grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    path = os.path.join(path_grey, id_)
    img = imread(path, as_gray = True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    grey[n] = img

mask_list = sorted(list_only_images(mask_path,extensions))
mask = np.zeros((len(mask_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(mask_list):
    path = os.path.join(mask_path, id_)
    img = imread(path,as_gray = True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask[n] = img

grey_flat = grey.flatten()

mask_flat = mask.flatten()

mask_flat=np.where(mask_flat>100,1,0)
grey_flat=grey_flat/255

zeros = np.count_nonzero(mask_flat == 0)
ones = np.count_nonzero(mask_flat == 1)


print("MASK TYPE ",mask_flat)
print("GREY TYPE ",grey_flat)



for elem in mask_flat:
    if elem!=0 and elem!=1:
        print(elem)

fp, tp, thr = metrics.roc_curve(mask_flat,grey_flat)
roc_auc = metrics.roc_auc_score(mask_flat, grey_flat) #  shape (n_samples,)

#plt.plot(fp,tp)
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

recall_list = list()
precision_list = list()
fallout_list = list()
accuracy_list =  list()
f1_list = list()

#thr_list = [100,150]
max_grey = np.max(grey_flat)

for thr in tqdm(range(0, 255)):  # range(1, max_grey)
    thr=thr/255

    bw_flat = np.where(grey_flat>thr, 1, 0)

    TN, FP, FN, TP = metrics.confusion_matrix(mask_flat,bw_flat).ravel()

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    fallout = FP/(FP+TN)
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    f1 = 2*((precision*recall)/(precision+recall))

    recall_list.append(recall)
    precision_list.append(precision)
    fallout_list.append(fallout)
    accuracy_list.append(accuracy)
    f1_list.append(f1)


thr_best = np.argmax(f1_list)

acc_best = accuracy_list[thr_best]
prec_best = precision_list[thr_best]
rec_best = recall_list[thr_best]
fallout_best = fallout_list[thr_best]
f1_best = f1_list[thr_best]

save_path = os.path.join(run_path, "metrics")

try:
    os.mkdir(save_path)
except:
    print("")

run_name = os.path.basename(os.path.normpath(run_path))


data = {'Run': [run_name], 'thr': [thr_best], 'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best], 'auc': [roc_auc]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics.xlsx'))



 

