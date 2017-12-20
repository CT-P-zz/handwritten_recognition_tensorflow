import os
import sys
import hw_utils
from PIL import Image
import pandas as pd
import numpy as np 
import cv2
from PIL import Image
#hardcoded part, please change to yours
def convert_images():

    lstDir = os.walk(str("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_word"))

    
    df = pd.read_csv(str("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/CSV/IAM test/test_tickets.csv"), sep=",",index_col="index")
    df = df.loc[:, ['image']]
    lstIm = df.as_matrix()
    

    for file in os.listdir("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_words"):
        
                  
        (name, ext) = os.path.splitext(file)
       
        if name in lstIm:
          
            img = cv2.imread("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_words/"+str(name+ext))

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
            cv2.imwrite("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_wordsBW/"+str(name+ext),thresh)
    
            hw_utils.scale_invert("C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_wordsBW/"+str(name+ext),"C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Images/"+str(name+ext),64,1024)

if __name__ == '__main__':
    convert_images()