#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/SqueezeNet/RistrettoDemo/quantized.prototxt \
	--weights=models/SqueezeNet/RistrettoDemo/squeezenet_finetuned.caffemodel \
	--iterations=200

#--gpu=0 --iterations=2000
