# -*- coding: utf-8 -*-

import os
import wave
import difflib
import numpy as np
import torch
import onnxruntime as rt
import onnx
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs

import cv2
from copy import deepcopy
from pathlib import Path
from typing import List, Dict
import time
from torch import nn

import sys

def sess_run(model, input_data, save_bname=None):
    sess = rt.InferenceSession(model)
    input_name = []
    for n in sess.get_inputs():
        input_name.append(n.name)
    print("input name", input_name)
    output_name = []
    for n in sess.get_outputs():
        output_name.append(n.name)
    print("output name", output_name)

    res = sess.run(None, {input_name[i]: input_data[i]  for i in range(len(input_name))})

    return res

if __name__ == '__main__':

    bgr_img = cv2.imread("test.jpg",0)
    bgr_img = cv2.resize(bgr_img,(320,240), interpolation=cv2.INTER_AREA)
    data = (bgr_img/255.).astype(np.float32)[None, None]
    print('data', data)

    onx_data0 = sess_run('SuperPoint.onnx', [data])
    print(onx_data0[0].shape)
    print(onx_data0[1].shape)
    print(onx_data0[2].shape)
    print(onx_data0)

    onx_data1 = sess_run('SuperPoint.onnx', [data])
    #print(onx_data1[0].shape)
    #print(onx_data1[1].shape)
    #print(onx_data1[2].shape)
    #print(onx_data1)

    datas = [np.array([x]) for x in onx_data0+onx_data1]
    res = sess_run('SuperMatch.onnx', datas)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)
    print(res[3].shape)
    print(res)
