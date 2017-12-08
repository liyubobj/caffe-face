#!/usr/bin/python

import os
import time
import cPickle
import datetime
import logging
import optparse
import numpy as np
from PIL import Image
from skimage import transform
import cStringIO as StringIO
import urllib
import cv2

import caffe

REPO_DIRNAME = "."

class FaceSimilarity(object):

    def __init__(self):
        logging.info('Loading net and associated files...')
        self.model_def_file = '{}/model/face_deploy.prototxt'.format(REPO_DIRNAME)
        self.model_file = '{}/model/face_model.caffemodel'.format(REPO_DIRNAME)

        # set GPU mode
        caffe.set_device(0)
        caffe.set_mode_gpu()

        self.net = caffe.Net(self.model_def_file, self.model_file, caffe.TEST)

        # load face image, and align to 112 X 96
        self.shape = [112, 96];

        #coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
        #        51.6963, 51.5014, 71.7366, 92.3655, 92.2041];

        #facial5points = [105.8306, 147.9323, 121.3533, 106.1169, 144.3622; ...
        #        109.8005, 112.5533, 139.1172, 155.6359, 156.3451];

        src = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
        #dst = mtcnn_landmark.astype(np.float32)

        dst = np.array([[105.8306, 109.8005], [147.9323, 112.5533], [121.3533, 139.1172], [106.1169, 155.6359], [144.3622, 156.3451]], dtype=np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        self.M = tform.params[0:2,:]

    def crop_image(self, img):
        return cv2.warpAffine(img, self.M, (self.shape[1], self.shape[0]), borderValue = 0.0)

    def abstract_deep_feature(self, cropImg):
        print cropImg.shape
        if cropImg.shape[2] < 3:
            cropImg[:,:,2] = cropImg[:,:,1];
            cropImg[:,:,3] = cropImg[:,:,1];

        cropImg = (cropImg.astype(np.float32) - 127.5) * 0.0078125
        cropImg = np.transpose(cropImg, (2,0,1))
        print cropImg.shape
        self.net.blobs['data'].data[...] = cropImg

        return self.net.forward()["fc5"]
        #cropImg = single(cropImg);
        #cropImg = (cropImg - 127.5)/128;
        #cropImg = permute(cropImg, [2,1,3]);
        #cropImg = cropImg(:,:,[3,2,1]);

        #cropImg_(:,:,1) = flipud(cropImg(:,:,1));
        #cropImg_(:,:,2) = flipud(cropImg(:,:,2));
        #cropImg_(:,:,3) = flipud(cropImg(:,:,3));

        # extract deep feature
        #res = self.net.forward({cropImg});
        #res_ = self.net.forward({cropImg_});
        #deepfeature = [res{1}; res_{1}];

	#def classify(src_img_file, dst_img_file):
    #    src_img = cv2.imread(src_img_file)
    #    src_aligned = self.align_face(src_img)
    #    dst_img = cv2.imread(dst_img_file)
    #    dst_aligned = self.align_face(dst_img)

	#cv2.imwrite('{}/Jennifer_Aniston_0016_wraped.jpg'.format(REPO_DIRNAME), warped)


if __name__ == '__main__':
    face = FaceSimilarity()
    img_file = '{}/Jennifer_Aniston_0016_wraped.jpg'.format(REPO_DIRNAME)
    img = cv2.imread(img_file)
    crop_img = face.crop_image(img)
    feature = face.abstract_deep_feature(crop_img)
    print feature.shape
