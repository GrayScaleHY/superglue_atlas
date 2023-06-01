#!/bin/bash

atc --framework=5 \
	--model=SuperMatch.onnx \
	--input_shape_range="kpts0:[1,1~2048,2];scores0:[1,1~2048];descriptors0:[1,256,1~2048];kpts1:[1,1~2048,2];scores1:[1,1~2048];descriptors1:[1,256,1~2048]" \
	--input_format=ND \
	--output=SuperMatch \
	--soc_version=Ascend310P3 \
	--precision_mode=allow_mix_precision \
	--log=error