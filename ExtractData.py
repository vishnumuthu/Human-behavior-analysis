
# coding: utf-8

import cv2 as cv 
import numpy as np
import scipy
from PIL import Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt
from scipy.ndimage.filters import gaussian_filter
import glob
import csv

def getImageCamera(oriImg, net,filename):
    
    #ret, oriImg = cameraCV.read()
    #oriImg = cv.resize(oriImg, (320,240))
    param, model = config_reader()
    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    multiplier = [multiplier[0], multiplier[0]*1.1, multiplier[0],multiplier[0]]
    for m in range(len(multiplier)-2):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])

        net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        start_time = time.time()
        output_blobs = net.forward()

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
        	
        paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(19-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],            [1,16], [16,18], [3,17], [6,18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],           [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],           [55,56], [37,38], [45,46]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm!=0:
                        vec = np.divide(vec, norm)
                    
                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),                                np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]                                   for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]                                   for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # visualize
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')

    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv.circle(oriImg, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    stickwidth = 4
    if len(subset) == 0:
	SImg = np.zeros((20,20))
	print(filename)
    for n in range(len(subset)):
    	Xset = []
    	Yset = []
	SImg = np.zeros((20,20))
    	for i in range(17):
        
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            #cur_oriImg = oriImg.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
	    cv.line(oriImg, (int(Y[0]), int(X[0])), (int(Y[1]), int(X[1])), colors[i], 2)
            Xset.append(X)
	    Yset.append(Y)
 
	Xmax = np.amax(Xset)
	Ymax = np.amax(Yset)
	Xmin = np.amin(Xset)
	Ymin = np.amin(Yset)
	Xdiff = Xmax - Xmin
	Ydiff = Ymax - Ymin
	Xset = Xset - Xmin + 1
	Yset = Yset - Ymin + 1
	if Xdiff > Ydiff:
		Smove = (Xdiff - Ydiff)/2
		Side = Xdiff
		Yset = Yset + Smove
	else:
		Smove = (Ydiff - Xdiff)/2
		Side = Ydiff
		Xset = Xset + Smove
	Xset = np.around(Xset)
	Yset = np.around(Yset)
	Xset = np.around(Xset*(19/Side)) + 1
	Yset = np.around(Yset*(19/Side)) + 1
	Xset = Xset.astype(int)
	Yset = Yset.astype(int)

	for j in range(Xset.shape[0]):
	    cv.line(SImg, (Yset[j][0], Xset[j][0]), (Yset[j][1], Xset[j][1]), colors[j], 2)
    data = []
    for j in SImg:
	data.extend(j)

    cv.imshow('Result1',SImg)
    cv.waitKey(1)

    cv.imshow('Result',oriImg)
    cv.waitKey(1)

    return data
    """cv.imshow('Result1',SImg)
    if cv.waitKey(1) & 0xFF==ord('q'):
	quit()
    cv.imshow('Result',oriImg)
    if cv.waitKey(1) & 0xFF==ord('q'):
        quit()
    return"""


####################################################################################################

# Initializing model
param, model = config_reader()
caffe.set_mode_gpu()
caffe.set_device(param['GPUdeviceNumber']) # set to your device!
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

#cameraCV = cv.VideoCapture(0)
#while(True):
finaldata = []
for filename in glob.glob('/<modal path>/<modal name>/*.jpg'):
    #print(filename)
    im = cv.imread(filename)
    data = getImageCamera( im, net ,filename)
    finaldata.append(data)
print(np.shape(finaldata))
f = open('<modal name>.csv','w')
for i in finaldata:
	csv.writer(f).writerow(i)
f.close 









