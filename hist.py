#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import cv2

h_dict = {}
co_ord = {}
h_list = {}

def hist(img, i):
	#h_col = {}
	h_col = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
	h_col = cv2.normalize(h_col).flatten()
	#index[filename] = hist
	#color = ('b','g','r')
	#for channel,col in enumerate(color):
	    #histr = cv2.calcHist([img],[channel],None,[16],[0,256])
	    #h_col[col] = histr
	    #plt.plot(histr,color = col)
	    #plt.xlim([0,16])
	plt.plot(h_col)
	#h_list[i] = h_col
	plt.title('Histogram for color scale picture')
	plt.show()
	return h_col

f = '1.JPG'
f2 = '2.JPG'
i = 0
img = cv2.imread('1.JPG')
img2 = cv2.imread('2.JPG')
bbox = [0, 0, 0, 0]
bbox2 = [0, 0, 0, 0]
bbox[0] = 100
bbox[1] = 100
bbox[2] = 300
bbox[3] = 200
bbox2[0] = 120
bbox2[1] = 120
bbox2[2] = 300
bbox2[3] = 200
im = img[bbox[0]:bbox[0]+bbox[3], bbox[1]:bbox[1]+bbox[2]]
im2 = img2[bbox2[0]:bbox2[0]+bbox2[3], bbox2[1]:bbox2[1]+bbox2[2]]
cv2.imshow('res',im)
cv2.waitKey(5)

H1 = hist(im, i)
h_list[i] = hist(im, i)
if img in h_dict.keys():
	h_dict[f].append(h_list)
	co_ord[i].append(bbox)
	i += 1
else:
	h_dict[f] = h_list
	co_ord[i] = bbox
	i += 1


band = 5
for key in co_ord.keys():
	val = co_ord[key]
	midx1 = val[0] + (val[3]/2)
	midy1 = val[1] + (val[2]/2)
	midx2 = bbox2[0] + (bbox2[3]/2)
	midy2 = bbox2[1] + (bbox2[2]/2)
	if((midx1-band<=midx2 and midx2<=midx1+band) and (midy1-band<=midy2 and midy2<=midy1+band)):
		c = 0
	else:
		H2 = hist(im2, i)
		h_list = hist(im2, i)
		if img2 in h_dict.keys():
			h_dict[f2].append(h_list)
			co_ord[i].append(bbox2)
			i += 1
		else:
			h_dict[f2] = h_list
			co_ord[i] = bbox2
			i += 1
		cv2.imshow('res',im2)
		cv2.waitKey(5)
		print(co_ord)
	p = cv2.compareHist(H1, H2, cv2.cv.CV_COMP_BHATTACHARYYA)
	print(p)
	
