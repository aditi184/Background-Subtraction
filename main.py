""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args

def train_dev_split(args):
    # find out the evaluation frames (assuming they will be at the end of the video)
    eval_frames = open(args.eval_frames).readlines()[0].split(' ')
    eval_frame_start = int(eval_frames[0])-1
    eval_frame_end = int(eval_frames[1])

    # complete video sequence
    filenames = os.listdir(args.inp_path)
    filenames.sort()

    # split the video sequence into train and dev set (as per eval frames given)
    train_data = filenames[0 : eval_frame_start] 
    dev_data = filenames[eval_frame_start : eval_frame_end]
    print("train frames:", train_data[0], "to", train_data[-1])
    print("eval frames:", dev_data[0], "to", dev_data[-1])

    return train_data, dev_data

def baseline_bgs(args):
    train_data, dev_data = train_dev_split(args)
    
    #Hyperparams
    history = 90
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold=varThreshold, detectShadows=False)
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def histogram_equalization(img):
    # Perform Histogram Equalization to counter illumination change problem
    # convert the color scheme to YCrCb which separate the intensity/brightness information of image separately unlike rgb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    # perform histogram equalization on the Y-channel of this image
    # for this, we use CLAGE's algorithm (that performs piecewise histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    img_ycrcb[:,:,0] = clahe.apply(img_ycrcb[:,:,0])
    
    # convert back to bgr
    img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img

def adaptive_thresholding(img):
    channels = []
    for channel in cv2.split(img):
        channel_bg = cv2.medianBlur(cv2.dilate(channel, np.ones((3,3))), 25)
        channel_edges = cv2.absdiff(channel, channel_bg)
        channel_edges = cv2.normalize(channel_edges, dst = None, norm_type = cv2.NORM_MINMAX, alpha = 0, beta = 255)
        channels.append(channel_edges)
    return cv2.merge(channels)

def illumination_bgs(args):
    train_data, dev_data = train_dev_split(args)
    
    # hyper params
    history = 50
    varThreshold = 300
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # read the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) 
        imgs.append(img)

        img = histogram_equalization(img)
        img = adaptive_thresholding(img)
        _ = background_model.apply(img, learningRate=learningRate)
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        img = histogram_equalization(img)
        img = adaptive_thresholding(img)
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        pred_mask = cv2.resize(pred_mask, (320, 240))
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def jitter_bgs(args):
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 250
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def dynamic_bgs(args):
    train_data, dev_data = train_dev_split(args)

    # hyper params
    history = 250
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # read the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) 
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
        
    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)

        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def ptz_bgs(args):
    # IMPORTANT - eval_frames.txt is not present in data but it is named as temporalROI.txt | This is handled by giving correct path in the arguments
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 200
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)

        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def main(args):
    if args.category not in "bijmp": # error in main.py -> earlier it was bijdp
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)