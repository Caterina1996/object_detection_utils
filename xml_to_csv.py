"""
Usage:
  # From auxiliary folder (object-detection)
  # Create csv files:
  python3 xml_to_csv.py
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Path to the folder')
parsed_args = parser.parse_args()
folder = parsed_args.folder


def xml_to_csv(path):
    xml_list = []
    for xml_path in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        xml_file = os.path.split(xml_path)[1]
        xml_name = os.path.splitext(xml_file)[0]
        extension = os.path.splitext(root.find('filename').text)[1]
        im_file = xml_name + extension

        for member in root.findall('object'):
            if member[0].text != "halimeda":
                print("salgo con: " + str(member[0].text))

            value = (im_file,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():

    for directory in ['train', 'test']:
        image_path = os.path.join(folder, 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(folder + '/images/{}_labels.csv'.format(directory), index=None)

        print('Successfully converted xml to csv.')


main()





