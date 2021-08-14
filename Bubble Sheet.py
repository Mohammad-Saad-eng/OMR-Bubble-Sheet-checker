#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours


# In[53]:


image = cv2.imread(r"D:\OpenCV\17 Days Crash Course\OMR\test_10.jpg")
image1 = image.copy()

heightImg = 600
widthImg  = 480

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 75,200)

cnts, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key = cv2.contourArea)

biggest = np.array([])
for i in cnts:
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    biggest = approx

biggest1 = biggest.reshape(4,2)
rect = np.zeros((4,2), dtype='float32')

s = biggest1.sum(axis=1)
rect[0] = biggest1[np.argmin(s)]
rect[2] = biggest1[np.argmax(s)]

dif = np.diff(biggest1, axis=1)
rect[1] = biggest1[np.argmin(dif)]
rect[3] = biggest1[np.argmax(dif)]

cv2.drawContours(image, [cnt], -1, (255,0,0),3)

#pts1 = np.float32([biggest[1],biggest[0],biggest[2],biggest[3]])
pts2 = np.float32([[0, 0],[widthImg, 0],[widthImg, heightImg], [0, heightImg]])
M = cv2.getPerspectiveTransform(rect, pts2)
pers = cv2.warpPerspective(image1, M, (widthImg, heightImg))

warped = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
    
image2 = pers.copy()
cv2.drawContours(image2, questionCnts, -1, (255,0,0),3)

plt.figure(figsize=[15,15])
plt.subplot(221);plt.imshow(image)
plt.subplot(222);plt.imshow(pers[...,::-1])
plt.subplot(223);plt.imshow(thresh,cmap='gray')
plt.subplot(224);plt.imshow(image2)


# In[54]:


questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
correct = 0
paper = pers.copy()
not_at = []
over_at = []

for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None
    l = 0
    marked = 0
    
    for (j,c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        
        if marked == 0:
            if (bubbled is None or total > bubbled[0]) and total > 900:
                bubbled = (total, j)
                l += 1
                marked += 1
                
        elif marked == 1:
            if bubbled is None or total > bubbled[0] or total > 900:
                marked += 1
                
        elif marked > 1:
            bubbled = (total, -1)
                      
    if l == 0:
        not_at.append(q)
    
    if marked > 1:
        over_at.append(q)
        bubbled = (total, -1)
    
    if l != 0 and marked <= 1:         
        color = (0,0,255)
        k = ANSWER_KEY[q]
        
        if k == bubbled[1]:
            color = (0,255,0)
            correct += 1
        cv2.drawContours(paper, [cnts[k]], -1, color,3)
 
score = (correct / 5.0) * 100
if score < 50.0:
    cv2.putText(paper, "{:.2f}%".format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(paper, "Give up on your dreams and die", (40,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
else:
    cv2.putText(paper, "{:.2f}%".format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
if len(not_at) != 0 or len(over_at) != 0:
    if len(not_at) != 0:
        for i in not_at:
            print("MCQ Number {} is unattempted".format(i+1))
    if len(over_at) != 0:
        for i in over_at:
            print("MCQ Number {} is over attempted".format(i+1))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Paper")
plt.subplot(122);plt.imshow(paper[...,::-1]);plt.title("Checked Paper")


# In[56]:


image2 = cv2.imread(r"D:\OpenCV\17 Days Crash Course\OMR\test_01.png")
image1 = image2.copy()

heightImg = 600
widthImg  = 480

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 75,200)

cnts, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key = cv2.contourArea)

biggest = np.array([])
for i in cnts:
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    biggest = approx

biggest1 = biggest.reshape(4,2)
rect = np.zeros((4,2), dtype='float32')

s = biggest1.sum(axis=1)
rect[0] = biggest1[np.argmin(s)]
rect[2] = biggest1[np.argmax(s)]

dif = np.diff(biggest1, axis=1)
rect[1] = biggest1[np.argmin(dif)]
rect[3] = biggest1[np.argmax(dif)]

cv2.drawContours(image2, [cnt], -1, (255,0,0),3)

#pts1 = np.float32([biggest[1],biggest[0],biggest[2],biggest[3]])
pts2 = np.float32([[0, 0],[widthImg, 0],[widthImg, heightImg], [0, heightImg]])
M = cv2.getPerspectiveTransform(rect, pts2)
pers = cv2.warpPerspective(image1, M, (widthImg, heightImg))

warped = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
    
image3 = pers.copy()
cv2.drawContours(image3, questionCnts, -1, (255,0,0),3)

plt.figure(figsize=[15,15])
plt.subplot(221);plt.imshow(image2)
plt.subplot(222);plt.imshow(pers[...,::-1])
plt.subplot(223);plt.imshow(thresh,cmap='gray')
plt.subplot(224);plt.imshow(image3)


# In[57]:


questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
correct = 0
paper1 = pers.copy()
not_at = []
over_at = []

for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None
    l = 0
    marked = 0
    
    for (j,c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        
        if marked == 0:
            if (bubbled is None or total > bubbled[0]) and total > 900:
                bubbled = (total, j)
                l += 1
                marked += 1
                
        elif marked == 1:
            if bubbled is None or total > bubbled[0] or total > 900:
                marked += 1
                
        elif marked > 1:
            bubbled = (total, -1)
                      
    if l == 0:
        not_at.append(q)
    
    if marked > 1:
        over_at.append(q)
        bubbled = (total, -1)
    
    if l != 0 and marked <= 1:         
        color = (0,0,255)
        k = ANSWER_KEY[q]
        
        if k == bubbled[1]:
            color = (0,255,0)
            correct += 1
        cv2.drawContours(paper1, [cnts[k]], -1, color,3)
 
score = (correct / 5.0) * 100
if score < 50:
    cv2.putText(paper1, "{:.2f}%".format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(paper1, "Give up on your dreams and die", (40,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
else:
    cv2.putText(paper1, "{:.2f}%".format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
if len(not_at) != 0 or len(over_at) != 0:
    if len(not_at) != 0:
        for i in not_at:
            print("MCQ Number {} is unattempted".format(i+1))
    if len(over_at) != 0:
        for i in over_at:
            print("MCQ Number {} is over attempted".format(i+1))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image2);plt.title("Paper")
plt.subplot(122);plt.imshow(paper1[...,::-1]);plt.title("Checked Paper")


# In[58]:


plt.figure(figsize=[15,15])
plt.subplot(221);plt.imshow(image);plt.title("Paper")
plt.subplot(222);plt.imshow(paper[...,::-1]);plt.title("Checked Paper")
plt.subplot(223);plt.imshow(image2);plt.title("Paper")
plt.subplot(224);plt.imshow(paper1[...,::-1]);plt.title("Checked Paper")


# In[ ]:




