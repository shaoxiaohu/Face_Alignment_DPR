#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np


def get_files(rootDir):
    list_dirs = os.walk(rootDir)
    file_lists = []
    for root, dirs, files in list_dirs:
        for f in files:
            file_lists.append(os.path.join(root, f))
    return file_lists

def load_rect(rect_path):

    fp = open(rect_path, "r")
    s_line = fp.readline()
    sub_str = s_line.split()
    face_rect = np.array([float(x) for x in sub_str])
    fp.close()

    return face_rect



