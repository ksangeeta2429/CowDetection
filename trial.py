#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from collections import defaultdict


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

h_dict = {}
co_ord = {}
wt = {}
img_point = defaultdict(list)

#'1.JPG','2.JPG','3.JPG','4.JPG','5.JPG','6.JPG','7.JPG','8.JPG','9.JPG','10.JPG', '11.JPG','12.JPG','13.JPG','14.JPG','15.JPG','16.JPG','17.JPG','18.JPG','19.JPG','20.JPG','21.JPG','22.JPG','23.JPG','24.JPG','25.JPG','26.JPG','27.JPG','28.JPG','29.JPG',
im_names = ['30.JPG','31.JPG','32.JPG','33.JPG','34.JPG','35.JPG','36.JPG','37.JPG','38.JPG','39.JPG','40.JPG']
count = 1
band = 50
#increase the threshold value for more flexibility with matches
limit = 0.55
#vanishing rate for updating weights
alpha = 0.1
#pobability at which detection window is ignored
prob = 0.7

def hist(img):
	h_col = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
	h_col = cv2.normalize(h_col).flatten()
	#plt.plot(h_col)
	#plt.title('Histogram for color scale picture')
	#plt.show()
	return h_col

def checkPrevious(im, class_name, dets, imgName, idx, thresh=0.7):
    """Find more potential bounding boxes using previous detections."""
    global count
    global limit
    global alpha
    global prob
    #print("CheckPrev.....")
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
	#print("No predictions....")
	temp = {}
	if imgName in img_point.keys():
		img_point[imgName].append(count)
	else:
		img_point[imgName] = [count]
	count += 1
	for key in co_ord.keys():
            temp[key] = co_ord[key]
    
    else:
	    bool_list = {}
	    temp = {}
	    for key in co_ord.keys():
		bool_list[key] = 0

	    for i in inds:
		bbox = dets[i, :4]
		#take a temp co-ordinate dictionary
		x1 = int(bbox[0])
		x2 = int(bbox[2])
		y1 = int(bbox[1])
		y2 = int(bbox[3])
		# point is [col, row, height, width]
		pt = [x1, y1, x2-x1, y2-y1]
		img = im[y1:y2,x1:x2,:]
		#cv2.imshow('res',img)
	
		#cv2.waitKey(50)
		#calculate the color histogram and save it in the dictionary with image file key
		#All histograms of predicted patches gets saved
		#plt.plot(hist(img))
		#plt.show()
		if imgName in img_point.keys():
			img_point[imgName].append(count)
		else:
			img_point[imgName] = [count]
		h_dict[count] = hist(img)
		co_ord[count] = pt
		#initial value of wt is 1
                wt[count] = 1
		count += 1

		#check the prediction in list of previous predictions
		#If there is a similar prediction in previous frame, ignore else add the point from past into the list of potential miss
		#When to update co_ord?
		for key in co_ord.keys():
			val = co_ord[key]
			midx1 = val[0] + int(val[2]/2)
			midy1 = val[1] + int(val[3]/2)
			midx2 = x1 + int((x2-x1)/2)
			midy2 = y1 + int((y2-y1)/2)
			#band is the value that can change results
			if((midx1-band<=midx2 and midx2<=midx1+band) and (midy1-band<=midy2 and midy2<=midy1+band)):
				c = 0
		                bool_list[key] = 1

	    for it in img_point[imgName]:
		bool_list[it] = 1

	    for key in co_ord.keys():
		if bool_list[key] == 0:
		    temp[key] = co_ord[key]

    #temp has all the missed points 
    #match the color histograms around those with the past ones
    if temp:
	    if(idx > 0):
		for k in temp.keys():
			val = temp[k]
			match_list = {}
			p_list = {}
			j = 0
			half_ht = int(val[2]/2)
			half_wt = int(val[3]/2)
			midx = val[0] + half_ht
			midy = val[1] + half_wt
			#print("Wt given:",half_wt*0.25)
			#print("Ht given: ",half_ht*0.25)
			t = img_point[im_names[idx-1]]
			H1 = h_dict[key]
			# match the color histograms of previous image in a square of side 10 around the mean in the new image
			# 10 is the value that can change things
		     	for y in (int(midy)-int(half_wt*0.25),int(midy)+int(half_wt*0.25)):
				for x in (int(midx)-int(half_ht*0.25),int(midx)+int(half_ht*0.25)):
					#print(y,x)
					im2 = im[y-half_wt:y+half_wt,x-half_ht:x+half_ht,:]
					#cv2.imshow('res',img)
					#cv2.waitKey(10)
					H2 = hist(im2)
					#store all the scores in match_list
					#comparison options: 
					#cv2.cv.CV_COMP_CORREL: Computes the correlation between the two histograms
					#cv2.cv.CV_COMP_CHISQR: Applies the Chi-Squared distance to the histograms
					#cv2.cv.CV_COMP_INTERSECT: Calculates the intersection between two histograms
					#cv2.cv.CV_COMP_BHATTACHARYYA: Bhattacharyya distance, used to measure the “overlap” between the two histograms
					match_list[j] = cv2.compareHist(H1, H2, cv2.cv.CV_COMP_BHATTACHARYYA)
					p_list[j] = [x-half_ht, y-half_wt, val[2], val[3]]
					j += 1

		     	#lesser value of p, better it is
			#find the mid-point that gave the closest match
			#print("Scores:")
			#print(match_list)
			minimum = min(match_list.items(), key=lambda x: x[1])
			#print(minimum[1])
			#value of limit can change results
			if(minimum[1]<limit and wt[k] > prob):
		                #potential miss so add the histogram in h_dict and add the point in co-ord
				point = p_list[minimum[0]]
				img2 = im[point[1]:point[1]+point[3], point[0]:point[0]+point[2]]
				if imgName in img_point.keys():
					#img_point[imgName].append(count)
					#print("Added ",k)
					img_point[imgName].append(k)
				else:
					#img_point[imgName] = [count]
					img_point[imgName] = [k]

				h_dict[k] = hist(img2)
				co_ord[k] = point
				wt[k] = wt[k] - (alpha*wt[k])
				#print("Weight: ",wt)
				#delete the previous key for this point
				to_edit = img_point[im_names[idx-1]]
				for iv in range(len(to_edit)):
					if(to_edit[iv] == k):
						to_edit[iv] = -1
				img_point[im_names[idx-1]] = to_edit
				#print(img_point)
						

		
    #print("New co-ords:")
    #print(co_ord)
    #remove the images other than recent one i.e. idx-1 and all corresponding co-ordinates too
    if(idx > 0):
    	#print(img_point)
    	to_remove = img_point[im_names[idx-1]]
	#print(to_remove)
	for value in to_remove:
		if(value in co_ord.keys()):
			del co_ord[value]
		if(value in h_dict.keys()):
			del h_dict[value]
		if(value in wt.keys()):
			del wt[value]

	del img_point[im_names[idx-1]]
        #print co_ord
	new_visualization(im, imgName, co_ord)
		
def new_visualization(im, imgName, co_ord):
    #print("New_Visualization")
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for key in co_ord.keys():
        bbox = co_ord[key]
        ax.add_patch(
		plt.Rectangle((bbox[0], bbox[1]),
				bbox[2],
				bbox[3], fill=False,
				edgecolor='red', linewidth=3.5)
		)
    ax.set_title(																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		'New Detections')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.DATA_DIR, 'hist4', imgName))
    plt.draw()

def vis_detections(im, class_name, dets, imgName, thresh=0.7):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #print(dets[:, -1])
    if len(inds) == 0:
        return
    #print("In vis_detections")
    img = im
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.DATA_DIR, 'trial3', imgName))
    plt.draw()
    plt.close()

def demo(net, image_name, idx):
    """Detect object classes in an image using pre-computed object proposals."""
    print("Starting Demo")
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'trial2', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    print(boxes)
    #print(scores.shape)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7 #0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	#try color histogram comparison to find extra boxes
	if(cls == 'cow'):
		checkPrevious(im, cls, dets, image_name, idx, thresh=CONF_THRESH)
        vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    idx = 0
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, idx)																				
	idx += 1

    #plt.show()
