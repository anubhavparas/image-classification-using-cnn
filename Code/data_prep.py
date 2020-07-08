import numpy as np
import math
import os 
from random import shuffle
import constants as CONST 
import cv2
import pickle

def get_size_statistics():
    heights = []
    widths = []
    img_count = 0
    DIR = CONST.TRAIN_DIR
    for img in os.listdir(CONST.TRAIN_DIR):
        path = os.path.join(DIR, img)
        data = cv2.imread(path)
        #data = np.array(Image.open(path))
        heights.append(data.shape[0])
        widths.append(data.shape[1])
        img_count += 1
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))

#get_size_statistics()


def label_img(name):
    word_label = name.split('.')[0]
    label = CONST.LABEL_MAP[word_label]
    label_arr = np.zeros(2)
    label_arr[label] = 1
    return label_arr


def prep_and_load_data():
    DIR = CONST.TRAIN_DIR
    data = []
    image_paths = os.listdir(DIR)
    shuffle(image_paths)
    count = 0
    for img_path in image_paths:
        label = label_img(img_path)
        path = os.path.join(DIR, img_path)
        image = cv2.imread(path)
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
        image = image.astype('float') / 255.0 
        data.append([image, label])
        count += 1
        print(count)
        if count == CONST.DATA_SIZE:
            break

    shuffle(data)

    #with open('train_data.pickle', 'wb') as train_d_file:
    #    pickle.dump(train_data, train_d_file)
    print(len(data))
    print('done')

    return data


if __name__ == "__main__":
    prep_and_load_data()
    



