import numpy as np
import cPickle
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def celeb_face_loader_64(main_path):

    file_list = glob.glob(main_path + "/*.jpg")
    file_list = file_list[:6000]
    image_set = np.zeros((len(file_list),3,64,64), dtype=np.float32)
    for idx in tqdm(range(len(file_list))):
        img = cv2.imread(file_list[idx])
        img = cv2.resize(img,(64,64))
        img = np.swapaxes(img,0,2)
        #img = np.float32(img/255.)
        image_set[idx,:,:,:] = img

    image_set = np.float32(image_set)
    image_set = (image_set - 127.5)/127.5
    return image_set
