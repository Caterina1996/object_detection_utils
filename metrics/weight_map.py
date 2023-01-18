import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
'''
call:
python weight_map.py --dir_im /images/test --path_txt /labels/test --path_out /final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/weight_maps

'''

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_im', help='im input directory.')
    parser.add_argument('--path_txt', help='txt input directory.')
    parser.add_argument('--path_out', help='im output directory.')
    parsed_args = parser.parse_args(sys.argv[1:])

    dir_im = parsed_args.dir_im
    path_txt = parsed_args.path_txt
    path_out = parsed_args.path_out

    max_bbox_size = 0

    for file in sorted(os.listdir(path_txt)):
        if re.search("\.(txt)$", file):  # if the file is a txt
            file_path = os.path.join(path_txt, file)
            instances = getInstances(file_path)
            for i, instance in enumerate(instances):
                box = getBoxFromInst(instance)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                bbox_size = (right-left)*(bottom-top)
                if bbox_size > max_bbox_size:
                    max_bbox_size = bbox_size

    for file in sorted(os.listdir(path_txt)):

        if re.search("\.(txt)$", file):  # if the file is a txt

            name, ext = os.path.splitext(file)
            path_im = os.path.join(dir_im, name + ".jpg")

            image = imageio.imread(path_im)  # read image
            weight_im = np.ones([image.shape[0], image.shape[1]], dtype=np.uint16)  # auxiliary black image

            file_path = os.path.join(path_txt, file)
            instances = getInstances(file_path)

            for i, instance in enumerate(instances):
                box = getBoxFromInst(instance)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

                bbox_size = (right-left)*(bottom-top)
                weight = int(max_bbox_size/bbox_size)

                for j in range(top, bottom):
                    for k in range(left, right):
                        weight_im[j, k] = weight

            # save_path = os.path.join(path_out, name + "_weight" + ".xlsx")
            # df = pd.DataFrame(weight_im)
            # df.to_excel(save_path, index=False)

main()
