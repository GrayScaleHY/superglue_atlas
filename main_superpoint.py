import torch
from superpoint_test import SuperPoint
# superpoint转onnx
input_data = torch.randn(1, 1, 240, 320)
default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
}
superpoint = SuperPoint(default_config)
superpoint.eval()

superpoint.load_state_dict(torch.load("superpoint.pth",map_location='cpu'))
torch.onnx.export(superpoint, input_data, "SuperPoint.onnx", opset_version=11, verbose=True)



