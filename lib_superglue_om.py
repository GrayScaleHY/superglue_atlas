import sys
sys.path.append("acllite")

import numpy as np
import time
import cv2
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from datetime import datetime
from test_onnx_all import correct_offset

def convert_coor(coor_ref, M):
    """
    使用偏移矩阵M计算参考坐标发生偏移后的对应坐标。
    args:
        coor_ref: 参考坐标
        M: 偏移矩阵
    return: 
        (x, y): 转换后的坐标
    """
    if M is None:
        return coor_ref

    M = np.array(M, dtype=float)
    
    assert M.shape == (2, 3) or M.shape == (3, 3), "shape of M is not match !"

    coor_ref = np.array(list(coor_ref) + [1], dtype=float)
    coor_tag = np.dot(M, coor_ref) # (2, 3)的转换矩阵直接通过相乘得到转换后坐标

    if M.shape == (3, 3): # Homo坐标系
        x = coor_tag[0] / coor_tag[2]; y = coor_tag[1] / coor_tag[2]
        coor_tag = np.array([x, y], dtype=float)

    return tuple(coor_tag.astype(int))


img_resize = (320, 240) # 送进网络的图片大小
point_max = 2048 # 提取特征点数上限
flen = 4 # float len
# ACL resource initialization
acl_resource = AclLiteResource(device_id=0)
acl_resource.init()
# load model
sp_out_inf = [point_max * 2 * flen, point_max * flen, 256 * point_max * flen]
sg_out_inf = [point_max * 2 * flen, point_max * 2 * flen, point_max * flen, point_max * flen]
superpoint_model1 = AclLiteModel("SuperPoint_240.om", output_info=sp_out_inf)
superpoint_model2 = AclLiteModel("SuperPoint_240.om", output_info=sp_out_inf)
superglue_model = AclLiteModel("SuperMatch_240.om", output_info=sg_out_inf)

def cut_superpoints(sp_feats):
    """
    将空的特征点删除。
    args:
        sp_feats: superpoint提取的结果，list[kpts, scores, feats]
    return:
        kpts: 特征点坐标shape: (1, ~, 2)
        scores: 特征点的置信度shape: (1, ~)
        feats: 特征描述子shape: (1, 256, ~)
    """
    kpts = sp_feats[0]
    scores = sp_feats[1]
    feats = sp_feats[2]

    kpts = kpts[[not np.all(kpts[i] == 0) for i in range(kpts.shape[0])], :]
    point_num = kpts.shape[0]
    scores = scores[:point_num]
    point_max = feats.shape[1]
    feats = feats.reshape(256 * point_max,)
    feats = feats[:256 * point_num]
    feats = feats.reshape(256, point_num)

    kpts = np.expand_dims(kpts, 0)
    scores = np.expand_dims(scores, 0)
    feats = np.expand_dims(feats, 0)

    return kpts, scores, feats

def superglue_registration(img_ref, img_tag):
    """
    使用superpoint+superglue模型进行图像配准。
    args:
        img_ref: 参考图
        img_tag: 待配准图
    return:
        M: 偏移矩阵
    """
    ## 求偏移矩阵的尺寸还原矩阵
    H_ref, W_ref = img_ref.shape[:2]
    H_tag, W_tag = img_tag.shape[:2]
    ref_rx = W_ref / img_resize[0]; ref_ry = H_ref / img_resize[1]
    tag_rx = W_tag / img_resize[0]; tag_ry = H_tag / img_resize[1]
    M_ref = np.diag(np.array([1/ref_rx, 1/ref_ry, 1]))
    M_tag = np.diag(np.array([tag_rx, tag_ry, 0]))

    ## 图片前处理，rgb2gray & resize
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    if len(img_tag.shape) == 3:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)

    img_ref = cv2.resize(img_ref, img_resize, interpolation=cv2.INTER_AREA)
    img_ref = (img_ref/255.).astype(np.float32)[None, None]
    img_tag = cv2.resize(img_tag,img_resize, interpolation=cv2.INTER_AREA)
    img_tag = (img_tag/255.).astype(np.float32)[None, None]

    ## superpoint提取特征点
    sp_ref = superpoint_model1.execute([img_ref,])
    sp_tag = superpoint_model2.execute([img_tag,])
    kpts_ref, scores_ref, feats_ref = cut_superpoints(sp_ref)
    kpts_tag, scores_tag, feats_tag = cut_superpoints(sp_tag)

    ## superglue特征匹配
    feats_pair = [kpts_ref,scores_ref,feats_ref,kpts_tag,scores_tag,feats_tag]
    matches = superglue_model.execute(feats_pair, mdl_type=const.MDL_DYNAMIC_SHAPE)
    matches = matches[0][0]

    ## 求偏移矩阵
    kpts0 = sp_ref[0][:len(matches)]
    kpts1 = sp_tag[0][:len(matches)]
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts0) < 5:
        return None
    
    M, mask = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5)

    ## 偏移矩阵还原
    M = np.row_stack((M, np.array([0,0,0])))
    M_T = M.T
    M_T = M_ref @ M_T @ M_tag
    M = M_T.T
    M = np.delete(M, 2 ,0)

    return M

if __name__ == '__main__':
    img_ref = cv2.imread("img_raw0.jpg")
    img_tag = cv2.imread("img_raw1.jpg")

    img_ref = cv2.resize(img_ref,(1280, 960), interpolation=cv2.INTER_AREA)
    img_tag = cv2.resize(img_tag,(640, 640), interpolation=cv2.INTER_AREA)

    line = [500, 400, 700 , 550]
    M = superglue_registration(img_ref, img_tag)
    print("M:", M)
    
    c1 = convert_coor((line[0], line[1]), M)
    c2 = convert_coor((line[2], line[3]), M)

    cv2.line(img_ref, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    cv2.line(img_tag, c1,c2, (0, 255, 0), 2)

    cv2.imwrite("img_raw_ref.jpg", img_ref)
    cv2.imwrite("img_raw_tag.jpg", img_tag)


    
    

 