import torch
from superpoint_test import SuperPoint
from superglue import SuperGlue
# 获取superpoint结果数据
input_data = torch.randn(1, 1, 240, 320)
default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
}
superpoint = SuperPoint(default_config)
superpoint.load_state_dict(torch.load("superpoint.pth",map_location='cpu'))



# supermatch 转 onnx
default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.9,
}

data0 = superpoint(input_data)
data1 = superpoint(input_data)

#data = {}
#data["image0"] = input_data
#data["image1"] = input_data
#data["keypoints0"] = data0["keypoints"][0].unsqueeze(0)
#data["keypoints1"] = data1["keypoints"][0].unsqueeze(0)
#data["scores0"] = data0["scores"][0].unsqueeze(0)
#data["scores1"] = data1["scores"][0].unsqueeze(0)
#data["descriptors0"] = data0["descriptors"][0].unsqueeze(0)
#data["descriptors1"] = data1["descriptors"][0].unsqueeze(0)

input0 = [data0["keypoints"][0].unsqueeze(0),
          data0["scores"][0].unsqueeze(0), data0["descriptors"][0].unsqueeze(0)]
input1 = [data1["keypoints"][0].unsqueeze(0),
          data1["scores"][0].unsqueeze(0), data1["descriptors"][0].unsqueeze(0)]

supermatch = SuperGlue(default_config)
supermatch.eval()

pth_ret = supermatch(input0, input1)
#print(out)

input_names = ['kpts0', 'scores0', 'descriptors0', 'kpts1', 'scores1', 'descriptors1']
dynamic_axes = {'kpts0': {1: "-1"}, 'scores0': {1: "-1"}, 'descriptors0': {2: "-1"}, 'kpts1': {1: "-1"}, 'scores1': {1: "-1"}, 'descriptors1': {2: "-1"}}

#torch.onnx.export(supermatch, data, "SuperMatch.onnx", opset_version=11,do_constant_folding=False)
torch.onnx.export(supermatch, (input0, input1), "SuperMatch.onnx", opset_version=12, input_names=input_names, dynamic_axes=dynamic_axes)