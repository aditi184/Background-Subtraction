import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-p', '--pred_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-g', '--gt_path', type=str, default='groundtruth', required=True, \
                                                        help="Path for the ground truth masks folder")
    args = parser.parse_args()
    return args


def binary_mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1==255,  mask2==255))
    union = mask1_area+mask2_area-intersection
    if union == 0: 
        # only happens if both masks are background with all zero values
        iou = 0, True
        # print("wrong masks are being evaluated")
    else:
        iou = intersection/union 
    return iou


def main(args):
    # Note: make sure to only generate masks for the evaluation frames mentioned in eval_frames.txt
    # Keep only the masks for eval frames in <pred_path> and not the background (all zero) frames.
    filenames = os.listdir(args.pred_path)
    ious = []
    wrong_mask_count = 0
    for filename in filenames:
        pred_mask = cv2.imread(os.path.join(args.pred_path, filename))
        gt_mask = cv2.imread(os.path.join(args.gt_path, filename))
        try:
            assert pred_mask.shape == gt_mask.shape
        except:
            print("masks either not read or are of different shapes")
        iou = binary_mask_iou(gt_mask, pred_mask)

        if type(iou) == tuple:
            # print(filename)
            iou = iou[0]
            wrong_mask_count += 1

        ious.append(iou)
    print("%i wrong masks encountered in the data out of total %i masks"%(wrong_mask_count, len(filenames)))
    print("mIOU: %.4f"%(sum(ious)/len(ious)))

if __name__ == "__main__":
    args = parse_args()
    main(args)