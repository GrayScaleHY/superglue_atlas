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

    onx_16_data0 = sess_run('SuperPoint_fp16.onnx', [data])
    print(onx_16_data0[0].shape)
    print(onx_16_data0[1].shape)
    print(onx_16_data0[2].shape)
    print(onx_16_data0)

    onx_16_data1 = sess_run('SuperPoint_fp16.onnx', [data])
    #print(onx_16_data1[0].shape)
    #print(onx_16_data1[1].shape)
    #print(onx_16_data1[2].shape)
    #print(onx_16_data1)

    onx_data0 = sess_run('SuperPoint.onnx', [data])
    onx_data1 = sess_run('SuperPoint.onnx', [data])

    datas_16 = [np.array([x]) for x in onx_16_data0+onx_16_data1]
    onx_res_16 = sess_run('SuperMatch.onnx', datas_16)
    print(onx_res_16[0].shape)
    print(onx_res_16[1].shape)
    print(onx_res_16[2].shape)
    print(onx_res_16[3].shape)
    print(onx_res_16)

    datas = [np.array([x]) for x in onx_data0+onx_data1]
    onx_res = sess_run('SuperMatch.onnx', datas)

    for onx_16_data0, onx_data0 in zip(onx_16_data0, onx_data0):
        print(f'dtype: SuperPoint fp16 = {onx_16_data0.dtype}, SuperPoint fp32 = {onx_data0.dtype}')
        print(f'shape: SuperPoint fp16 = {onx_16_data0.shape}, SuperPoint fp32 = {onx_data0.shape}')
        print(((onx_data0 - onx_16_data0) < 1e-1).all())

    for onx_res_16, onx_res in zip(onx_res_16, onx_res):
        print(f'dtype: SuperMatch fp16 = {onx_res_16.dtype}, SuperMatch fp32 = {onx_res.dtype}')
        print(f'shape: SuperMatch fp16 = {onx_res_16.shape}, SuperMatch fp32 = {onx_res.shape}')
        print(((onx_res - onx_res_16) < 1e-1).all())
