import torch
from superpoint_test import SuperPoint
from superglue import SuperGlue

import os
import wave
import difflib
import numpy as np
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

superpoint_default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
}

supermatch_default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.9,
}

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

    # 随机生成值
    input_data = torch.randn(1, 1, 240, 320)

    superpoint = SuperPoint(superpoint_default_config)
    superpoint.eval()
    superpoint.load_state_dict(torch.load("superpoint.pth",map_location='cpu'))

    data0 = superpoint(input_data)
    data1 = superpoint(input_data)

    input0 = [data0["keypoints"][0].unsqueeze(0),
              data0["scores"][0].unsqueeze(0), data0["descriptors"][0].unsqueeze(0)]
    input1 = [data1["keypoints"][0].unsqueeze(0),
              data1["scores"][0].unsqueeze(0), data1["descriptors"][0].unsqueeze(0)]

    supermatch = SuperGlue(supermatch_default_config)
    supermatch.eval()

    pth_ret = supermatch(input0, input1)

    onx_input_data = input_data.numpy()
    onx_data0 = sess_run('SuperPoint.onnx', [onx_input_data])
    onx_data1 = sess_run('SuperPoint.onnx', [onx_input_data])

    datas = [np.array([x]) for x in onx_data0+onx_data1]
    onx_ret = sess_run('SuperMatch.onnx', datas)
    for pth_res, onx_res in zip(pth_ret.values(), onx_ret):
        print(f'dtype: pth_res = {pth_res.dtype}, onx_res = {onx_res.dtype}')
        print(f'shape: pth_res = {pth_res.shape}, onx_res = {onx_res.shape}')
        print(((pth_res.detach().numpy() - onx_res) < 1e-5).all())