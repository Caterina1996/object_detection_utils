import os
import re
import glob
import sys
import argparse
import numpy as np
from scipy import ndimage
import scipy.misc
import pandas as pd

import imageio.v2 as imageio

'''
call:
python coverage.py --shape 1024 --path_txt ../halimeda/cthr/ --path_out ../halimeda/coverage --grid 500


python coverage.py --shape 1024 --path_txt  --path_out ../halimeda/coverage --grid 500

python /mnt/c/Users/haddo/DL_stack/object_detection_utils/metrics/coverage.py --shape 1024 \
            --path_txt /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/{}/inference_val/labels/ \
                    --path_out /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/{}/inference_val/coverage/ --grid 500

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
            print("this is gt!!!")
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
    parser.add_argument('--shape', help='images shape.', type=int)   
    parser.add_argument('--path_txt', help='txt input directory.')
    parser.add_argument('--path_out', help='im output directory.')
    parser.add_argument('--grid', default=0, help='grid AxA.')
    parsed_args = parser.parse_args(sys.argv[1:])

    shape = parsed_args.shape
    path_txt = parsed_args.path_txt
    path_out = parsed_args.path_out
    grid = int(parsed_args.grid)

    try:
        os.mkdir(path_out)
    except:
        print("")
    
    test_cases = list()
    cov_pix_list = list()
    cov_grid_list = list()

    for file in sorted(os.listdir(path_txt)):
        if re.search("\.(txt)$", file):  # if the file is a txt
            name, ext = os.path.splitext(file)
            test_cases.append(name)
            aux_im = np.zeros([shape, shape], dtype=np.uint8)  # auxiliary black image   
            file_path = os.path.join(path_txt, file)
            instances = getInstances(file_path)

            for i, instance in enumerate(instances):
                box = getBoxFromInst(instance)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

                for j in range(top, bottom):
                    for k in range(left, right):
                        #Check this!!
                        if len(instance)==6:
                            aux_im[j, k] = int(255*instance[1])
                        elif len(instance)==5:
                            aux_im[j, k] = int(255)

            cov_pix = (np.sum(aux_im != 0)/np.size(aux_im))*100
            cov_pix_list.append(cov_pix)

            if grid > 0:
                count = 0
                total = grid*grid
                step_h = shape/grid  
                step_w = shape/grid   
                index_h = list()
                index_w = list()

                for i in range(1, grid):
                    index_h.append(int(step_h*i))
                    index_w.append(int(step_w*i))

                split1 = np.array_split(aux_im, index_h)

                for sp1 in enumerate(split1):
                    split2 = np.array_split(sp1[1], index_w, axis=1)

                    for sp2 in enumerate(split2):
                        if np.sum(sp2[1] != 0) > 0:
                            count = count+1

                cov_grid = (count/total)*100
                cov_grid_list.append(cov_grid)

            save_path = os.path.join(path_out, name + "_cov" + ".jpg")
            imageio.imwrite(save_path, aux_im)  # generate image file

            # save spine results on csv
            header = ['cov_pix', 'cov_grid']
            cov_csv = ({header[0]: cov_pix_list, header[1]: cov_grid_list})
            df = pd.DataFrame.from_records(cov_csv, index=test_cases)
            df.to_csv(path_out + "/coverage_"+str(grid)+".csv")


main()
