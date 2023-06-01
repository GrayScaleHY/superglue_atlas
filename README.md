### superglue_atlas
1. 使用main_supermatch.py将superglue.pth模型转换成superglue.onnx模型  
2. 使用main_superpoint.py将superpoint.pth模型转换成superpoint.onnx模型
3. 使用atc_supermatch.sh将superglue.onnx模型转换成superglue.om模型
4. 使用atc_superpoint.sh将superpoint.onnx模型转换成superpoint.om模型
5. 使用lib_superglue_om.py进行模型推理

注意：转onnx，pytorch要用1.9.0   ```pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0```

参考代码https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SuperGlue_with_SuperPoint_for_Pytorch
