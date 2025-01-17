import os
import re
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

"""
call:

python eval_weights.py  --pred_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_pred \
                        --gt_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_gt \
                        --gt_label_path /mnt/c/Users/haddo/yolov5/datasets/halimeda/labels/test \
                        --run_name yolo_XL_hyp_high_lr2_a \
                        --save_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test \
                        --shape 1024
 """


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def getInstances(file):
    instances = list()
    fh1 = open(file, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = (splitLine[0])  # class
        if len(splitLine) == 5:
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            inst = (idClass, x, y, w, h)  
            bbox=yolo_to_xml_bbox([x, y, w, h], 1024, 1024)
            inst = (idClass, bbox[0], bbox[1], bbox[2], bbox[3])
        elif len(splitLine) == 6:
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            inst = (idClass, confidence, x, y, w, h)   
            bbox=yolo_to_xml_bbox([x, y, w, h], 1024, 1024)
            inst = (idClass, confidence, bbox[0], bbox[1], bbox[2], bbox[3])
        instances.append(inst)
    fh1.close()
    return instances


def getBoxFromInst(inst):
    if len(inst) == 5:
        box = (inst[1], inst[2], inst[3], inst[4])
    elif len(inst) == 6:
        box = (inst[2], inst[3], inst[4], inst[5])
    return box

def list_only_images(dir,extensions):
    images_list=[]
    for extension in extensions:
        images_list.extend(glob.glob(dir+extension))

    images_ids=[x.split("/")[-1] for x in images_list]

    print(images_ids,"\n")
    return images_ids




parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', help='Path to the run folder', type=str)
parser.add_argument('--gt_path', help='Path to the gt folder', type=str)
parser.add_argument('--run_name', help='Path to the gt folder', type=str)
parser.add_argument('--save_path', help='Path to the gt folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)

parsed_args = parser.parse_args()

pred_path = parsed_args.pred_path
gt_path = parsed_args.gt_path
run_name = parsed_args.run_name
save_path = parsed_args.save_path
shape = parsed_args.shape

IMG_WIDTH = shape
IMG_HEIGHT = shape

extensions=["*.jpg","*.JPG","*.png","*.PNG"]

#gt_list = sorted(os.listdir(gt_path))
gt_list=sorted(list_only_images(gt_path,extensions))

gts = np.zeros((len(gt_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

#pred_list = sorted(os.listdir(pred_path))
pred_list=sorted(list_only_images(pred_path,extensions))
preds = np.zeros((len(pred_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) # use gt_list len to avoid problems caused by csvs in preds

print("PRED: ",pred_list,"\n")

for n, name in enumerate(pred_list):
    if ".csv" not in name:
        path = os.path.join(pred_path, name)
        img = imread(path, as_gray = True)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        preds[n] = img
    else:
        print("ALTO ! UN CSV!!")


for n, name in enumerate(gt_list):
    if ".csv" not in name:
        path = os.path.join(gt_path, name)
        img = imread(path,as_gray = True)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        gts[n] = img
    else:
        print("ALTO ! UN CSV!!")

print("PRED LIST: ",pred_list,"\n")
print("GT LIST: ",gt_list,"\n")
print("LEN PREDS: ",len(preds),"\n")
print("LEN GTS: ",len(gts),"\n")

weights = np.ones((len(gt_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)

max_bbox_size = 0
for file in sorted(os.listdir(gt_label_path)):
    if re.search("\.(txt)$", file):  # if the file is a txt
        file_path = os.path.join(gt_label_path, file)
        instances = getInstances(file_path)
        for i, instance in enumerate(instances):
            box = getBoxFromInst(instance)
            (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            bbox_size = (right-left)*(bottom-top)
            if bbox_size > max_bbox_size:
                max_bbox_size = bbox_size

for idx, file in enumerate(sorted(os.listdir(gt_label_path))):
    if re.search("\.(txt)$", file):  # if the file is a txt
        file_path = os.path.join(gt_label_path, file)
        instances = getInstances(file_path)
        for i, instance in enumerate(instances):
            box = getBoxFromInst(instance)
            (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            bbox_size = (right-left)*(bottom-top)
            weight = int(max_bbox_size/bbox_size)

            for j in range(top, bottom):
                for k in range(left, right):
                    weights[idx, j, k] = weight

pred_flat = preds.flatten()
gt_flat = gts.flatten()
gt_flat = np.where(gt_flat>100, 1, 0)


recall_list = list()
precision_list = list()
fallout_list = list()
accuracy_list =  list()
f1_list = list()

#thr_list = [100,150]
max_grey = np.max(grey_flat)

for thr in tqdm(range(1, max_grey)):  # range(1, max_grey)

    pred_flat = np.where(pred_flat>thr, 1, 0)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in tqdm(range(np.shape(gt_flat)[0])):

        gt = gt_flat[i]
        pred = pred_flat[i]

        if gt == pred:
            if gt == 0:
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if gt == 0:
                fp = fp + 1
            else:
                fn = fn + 1

    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fall = fp/(fp+tn)
    f1 = 2*((prec*rec)/(prec+rec))


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

try:
    os.mkdir(save_path)
except:
    print("")

data = {'Run': [run_name],'thr': [thr_best],'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics_ss.xlsx'))



 

