import sys
sys.path.append("../../02_code/01_python/common/acllite")

import os
import numpy as np
import acl
import time
from PIL import Image
import cv2

import torch
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from datetime import datetime

def deal_superpoint_result(result_list, max_point_num):
    output1 = result_list[0]
    output2 = result_list[1]
    output3 = result_list[2]

    output1 = output1[[not np.all(output1[i] == 0) for i in range(output1.shape[0])], :]
    point_num = output1.shape[0]
    output2 = output2[:point_num]
    output3 = output3.reshape(256 * max_point_num,)
    output3 = output3[:256 * point_num]
    output3 = output3.reshape(256, point_num)

    print('output1.shape', output1.shape)
    print('output2.shape', output2.shape)
    print('output3.shape', output3.shape)
    #print('output1', output1)
    #print('output2', output2)
    #print('output3', output3)

    output1 = output1.reshape(1, point_num, 2)
    output2 = output2.reshape(1, point_num)
    output3 = output3.reshape(1, 256, point_num)

    return output1, output2, output3

def main():

    #ACL resource initialization
    acl_resource = AclLiteResource(device_id=0)
    acl_resource.init()

    max_point_num = 2048
    float_len = 4
    output_info1 = [max_point_num * 2 * float_len, max_point_num * float_len, 256 * max_point_num * float_len]
    output_info2 = [max_point_num * 2 * float_len, max_point_num * 2 * float_len, max_point_num * float_len, max_point_num * float_len]

    #load model
    model1 = AclLiteModel("SuperPoint.om", output_info=output_info1)
    model2 = AclLiteModel("SuperPoint.om", output_info=output_info1)
    model3 = AclLiteModel("SuperMatch.om", output_info=output_info2)

    bgr_img = cv2.imread("test.jpg",0)
    bgr_img = cv2.resize(bgr_img,(320,240), interpolation=cv2.INTER_AREA)
    data = (bgr_img/255.).astype(np.float32)[None, None]

    #input_data = torch.randn(1, 1, 240, 320)
    #data = input_data.numpy()
    #data = data.astype(np.float32)[None, None]

    #print(data)
    #print(data.shape)
    #data.tofile("test.bin")

    StartTime = datetime.now()
    result_list1 = model1.execute([data,])
    EndTime = datetime.now()
    UseTime = (EndTime - StartTime)
    print("SuperPoint UseTime is %s" % UseTime)

    StartTime = datetime.now()
    result_list2 = model2.execute([data,])
    EndTime = datetime.now()
    UseTime = (EndTime - StartTime)
    print("SuperPoint UseTime is %s" % UseTime)

    print(result_list1)

    input1, input2, input3 = deal_superpoint_result(result_list1, max_point_num)
    input4, input5, input6 = deal_superpoint_result(result_list2, max_point_num)

    StartTime = datetime.now()
    result_list3 = model3.execute([input1,input2,input3,input4,input5,input6], mdl_type=const.MDL_DYNAMIC_SHAPE)
    EndTime = datetime.now()
    UseTime = (EndTime - StartTime)
    print("SuperMatch UseTime is %s" % UseTime)
    print(result_list3)

if __name__ == '__main__':
    main()
 