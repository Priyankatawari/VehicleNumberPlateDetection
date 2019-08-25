# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:48:08 2019

@author: Priyanka
"""


import json
import cv2
import urllib
import numpy as np
import imutils
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

dict_img = []
cnt = 0
with open("Indian_Number_plates.json") as f:
    for line in f:
        data = json.loads(line)
        imgUrl = data['content']
        xycoordinates = data['annotation'][0]['points']
        d = {"id": str(cnt), "url":imgUrl,"cod":xycoordinates}
        dict_img.append(d)
        print(dict_img)
        cnt +=1


index_of_image = 0

for i in range(len(dict_img)):
    try:
        req = urllib.request.urlopen(dict_img[i]['url'])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        coordinates = dict_img[i]['cod']
        print(coordinates)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
        edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
        val,edged= cv2.threshold(edged,23,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        
        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        
        if screenCnt is None:
            detected = 0
            print ("No contour detected")
        else:
            detected = 1
        
        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        
            # Masking the part other than the number plate        
            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(img,img,mask=mask)
            
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]
            
            if not os.path.exists("output/"):
                os.mkdir("output/")
            config = ("-l eng --oem 1 --psm 7")
            #Read the number plate
            text = pytesseract.image_to_string(Cropped, config=config)
            print("Detected Number is:",text)
            cv2.imwrite("output/{}_original_contour.jpg".format(str(i)),img)
            cv2.imwrite("output/{}_cropped_numberplate_[{}].jpg".format(str(i),str(text)),Cropped)
    except:
        pass