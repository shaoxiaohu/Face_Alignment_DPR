#!/usr/bin/env python3
# coding: utf-8

import cv2
import argparse
import numpy.linalg as LA

from util import *
from networks.DPRNet import DPRGLNet

def main(args):

    left_idx = [60]
    right_idx = [72]
    args.left_eye_idx = list(range(33,42)) + list(range(60,68)) + [96]
    args.right_eye_idx = list(range(42,51)) + list(range(68,76)) + [97]
    args.nose_idx = list(range(51,60))
    args.mouth_idx = list(range(76,96))

    align_errors = []
    lists = get_files("sample_images/WFLW")
    imglists = [name for name in lists
                if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.bmp')]

    # load model
    net = DPRGLNet()
    net.load_model(args)

    num = 0
    for img_fp in imglists:

        num += 1
        print('%s\n' % img_fp)

        # load the face image and the bounding box
        ori_img = cv2.imread(img_fp)
        rect_path = img_fp[0:-3] + 'rct'
        face_rect = load_rect(rect_path)

        # load ground truth points
        keypoints = []
        try:
            fp = open(img_fp[0:-3] + "txt", "r")
        except:
            fp = open(img_fp[0:-3] + "pts", "r")
        nn = int(fp.readline())
        for i in range(nn):
            s_line = fp.readline()
            sub_str = s_line.split()
            pts = np.array([float(x) for x in sub_str])
            keypoints.append(pts[0:2])
        fp.close()

        keypoints = np.array(keypoints)
        left_pupil = np.mean(keypoints[left_idx, :], 0)
        right_pupil = np.mean(keypoints[right_idx, :], 0)
        interocular_distance = LA.norm(left_pupil - right_pupil)

        # facial landmark prediction
        pre_pts = net.predict(ori_img, face_rect)

        # show the result
        if args.vis_rst:
            for i in range(args.pts_num):
                pts = pre_pts[i]
                cv2.circle(ori_img, (int(pts[0]), int(pts[1])), 2, (0, 255, 0), -1)
            cv2.imshow('result', ori_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        errors = np.linalg.norm(pre_pts - keypoints, axis=1)
        dsum = np.sum(errors)
        align_errors.append(dsum / (args.pts_num * interocular_distance))
        print('Error:', align_errors[-1])

    align_errors = np.array(align_errors)

    return align_errors


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='inference pipeline')
    parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--devices-id', default=1, type=int)
    parser.add_argument('--pts_num', default=98, type=int)
    parser.add_argument('--net', default='SmallMobileNetV2', type=str)
    parser.add_argument('--global-reinit-size', default=128, type=int)
    parser.add_argument('--global-reinit-path', default='models/98_points_global_reinit.pth.tar', type=str)
    parser.add_argument('--global-regress-size', default=256, type=int)
    parser.add_argument('--global-regress-path', default='models/98_points_global_regress.pth.tar', type=str)
    parser.add_argument('--global-mean-face-path', default='data/mean_98_face.npy', type=str)
    parser.add_argument('--local-reinit-size', default=196, type=int)
    parser.add_argument('--local-reinit-path', default='models/98_points_local_reinit.pth.tar', type=str)
    parser.add_argument('--local-regress-size', default=128, type=int)
    parser.add_argument('--local-regress-path', default='models/98_points_local_regress.pth.tar', type=str)
    parser.add_argument('--local-mean-face-path', default='data/mean_98_part_face.npy', type=str)
    parser.add_argument('--vis-rst', default=False, type=bool)
    args = parser.parse_args()
    align_errors = main(args)

