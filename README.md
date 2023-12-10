![alt text](https://sima.ai/wp-content/uploads/2022/01/SiMaAI_Logo_Trademarked_Digital_FullColor_Large.svg)
# README #

Welcome to SiMa.ai's Model Zoo! SiMa.ai offers Palete software to efficiently run machine learning software on it. This repository contains instructions on how to download some representative models, their quantization/compilation scripts and model binaries that can be used by SiMa.ai's MLSoC and its companion platforms. This model zoo is provided as an educational tool to familiarize with SiMa.ai's software toolchains and developer workflow.   

# Model List #

|  Model   |  Framework  |  Input Shape  |  FPS  |  Pretrained Model |  Compilation script |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|  alexnet   |  Onnx  |  1, 224, 224, 3  |  483  |  [Torchvision Link]( https://pytorch.org/vision/main/models/alexnet.html ) | [alexnet.py](scripts/alexnet.py) | 
|  bvlcalexnet-7  |  Onnx  |  1, 3, 224, 224  |  488  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/alexnet/model ) |  | 
|  caffenet-9  |  Onnx  |  1, 3, 224, 224  |  525  |  [ONNX Zoo Link]( https://github.com/onnx/onnx-tensorflow/wiki/ModelZoo-Status-(tag=v1.9.0) ) |  | 
|  ctdet_coco_dlav0_512   |  PyTorch  |  1, 3, 512, 512  |  332  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_ctdet_coco_dlav0_512.html ) | [ctdet_coco_dlav0_512.py](scripts/ctdet_coco_dlav0_512.py) | 
|  densenet-12  |  Onnx  |  1, 3, 224, 224  |  510  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/densenet-121/model/densenet-12.onnx ) |  | 
|  densenet-9  |  Onnx  |  1, 3, 224, 224  |  507  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/densenet-121/model ) |  | 
|  densenet121   |  PyTorch  |  1, 224, 224, 3  |  490  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet121.py](scripts/densenet121.py) | 
|  densenet121   |  Onnx  |  1, 3, 224, 224  |  480  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet121.py](scripts/densenet121.py) | 
|  densenet161   |  PyTorch  |  1, 224, 224, 3  |  302  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet161.py](scripts/densenet161.py) | 
|  densenet161   |  Onnx  |  1, 3, 224, 224  |  304  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet161.py](scripts/densenet161.py) | 
|  densenet169   |  PyTorch  |  1, 224, 224, 3  |  362  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet169.py](scripts/densenet169.py) | 
|  densenet169   |  Onnx  |  1, 3, 224, 224  |  374  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet169.py](scripts/densenet169.py) | 
|  densenet201   |  PyTorch  |  1, 224, 224, 3  |  275  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet201.py](scripts/densenet201.py) | 
|  densenet201   |  Onnx  |  1, 3, 224, 224  |  283  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet201.py](scripts/densenet201.py) | 
|  dla-34   |  PyTorch  |  1, 3, 224, 224  |  929  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_dla_34.html ) | [dla-34.py](scripts/dla-34.py) | 
|  efficientnet_b0   |  Onnx  |  1, 224, 224, 3  |  760  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b0.py](scripts/efficientnet_b0.py) | 
|  efficientnet_b1   |  Onnx  |  1, 224, 224, 3  |  592  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b1.py](scripts/efficientnet_b1.py) | 
|  efficientnet_b2   |  Onnx  |  1, 224, 224, 3  |  580  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b2.py](scripts/efficientnet_b2.py) | 
|  efficientnet_b3   |  Onnx  |  1, 224, 224, 3  |  492  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b3.py](scripts/efficientnet_b3.py) | 
|  efficientnet_b4   |  Onnx  |  1, 224, 224, 3  |  421  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b4.py](scripts/efficientnet_b4.py) | 
|  efficientnet_b5   |  Onnx  |  1, 224, 224, 3  |  317  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b5.py](scripts/efficientnet_b5.py) | 
|  efficientnet_b6   |  Onnx  |  1, 224, 224, 3  |  269  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b6.py](scripts/efficientnet_b6.py) | 
|  efficientnet_b7   |  Onnx  |  1, 224, 224, 3  |  208  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b7.py](scripts/efficientnet_b7.py) | 
|  efficientnet_v2_m   |  Onnx  |  1, 224, 224, 3  |  243  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_v2_m.py](scripts/efficientnet_v2_m.py) | 
|  efficientnet_v2_s   |  Onnx  |  1, 224, 224, 3  |  327  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_v2_s.py](scripts/efficientnet_v2_s.py) | 
|  efficientnet-b0_tensorflow   |  TensorFlow  |  1, 224, 224, 3  |  764  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_b0.html ) |  | 
|  efficientnet-b0-pytorch   |  PyTorch  |  1, 3, 224, 224  |  685  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_b0_pytorch.html ) | [efficientnet-b0-pytorch.py](scripts/efficientnet-b0-pytorch.py) | 
|  efficientnet-lite4-11  |  Onnx  |  1, 224, 224, 3  |  851  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4 ) |  | 
|  efficientnet-v2-b0   |  PyTorch  |  1, 3, 224, 224  |  646  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_v2_b0.html ) | [efficientnet-v2-b0.py](scripts/efficientnet-v2-b0.py) | 
|  efficientnet-v2-s   |  PyTorch  |  1, 224, 224, 3  |  319  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_v2_s.html ) | [efficientnet-v2-s.py](scripts/efficientnet-v2-s.py) | 
|  erfnet   |  PyTorch  |  1, 3, 208, 976  |  521  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_erfnet.html ) | [erfnet.py](scripts/erfnet.py) | 
|  googlenet-9  |  Onnx  |  1, 3, 224, 224  |  1101  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet/model ) |  | 
|  googlenet-v1-tf   |  TensorFlow  |  1, 224, 224, 3  |  1101  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_googlenet_v1_tf.html ) | [googlenet-v1-tf.py](scripts/googlenet-v1-tf.py) | 
|  higher-hrnet-w32-human-pose-estimation   |  PyTorch  |  1, 3, 512, 512  |  167  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_higher_hrnet_w32_human_pose_estimation.html ) | [higher-hrnet-w32-human-pose-estimation.py](scripts/higher-hrnet-w32-human-pose-estimation.py) | 
|  human-pose-estimation-3d-0001   |  PyTorch  |  1, 3, 256, 448  |  591  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_human_pose_estimation_3d_0001.html ) | [human-pose-estimation-3d-0001.py](scripts/human-pose-estimation-3d-0001.py) | 
|  mnasnet0_5   |  Onnx  |  1, 224, 224, 3  |  1373  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet0_5.py](scripts/mnasnet0_5.py) | 
|  mnasnet0_75   |  Onnx  |  1, 224, 224, 3  |  1130  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet0_75.py](scripts/mnasnet0_75.py) | 
|  mnasnet0-75   |  PyTorch  |  1, 3, 224, 224  |  1272  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet0_75.html ) |  | 
|  mnasnet1_0   |  Onnx  |  1, 224, 224, 3  |  1261  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet1_0.py](scripts/mnasnet1_0.py) | 
|  mnasnet1_3   |  Onnx  |  1, 224, 224, 3  |  1186  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet1_3.py](scripts/mnasnet1_3.py) | 
|  mnasnet1-0   |  PyTorch  |  1, 3, 224, 224  |  1296  |  [Torchvision Link]( https://pytorch.org/vision/0.12/_modules/torchvision/models/mnasnet.html ) |  | 
|  mnasnet1-3   |  PyTorch  |  1, 3, 224, 224  |  1143  |  [Torchvision Link]( https://pytorch.org/vision/0.12/_modules/torchvision/models/mnasnet.html ) |  | 
|  mnist-8  |  Onnx  |  1, 1, 28, 28  |  2279  |  [ONNX Zoo Link]( https://github.com/onnx/models ) |  | 
|  mobilenet_v2   |  Onnx  |  1, 224, 224, 3  |  1267  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mobilenetv2.html ) | [mobilenet_v2.py](scripts/mobilenet_v2.py) | 
|  mobilenet-v1-0.25-128   |  TensorFlow  |  1, 128, 128, 3  |  1751  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v1_0_25_128.html ) | [mobilenet-v1-0.25-128.py](scripts/mobilenet-v1-0.25-128.py) | 
|  mobilenet-v1-1.0-224-tf   |  TensorFlow  |  1, 224, 224, 3  |  1420  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v1_1_0_224_tf.html ) | [mobilenet-v1-1.0-224-tf.py](scripts/mobilenet-v1-1.0-224-tf.py) | 
|  mobilenet-v2-1.0-224   |  TensorFlow  |  1, 224, 224, 3  |  1363  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v2_1_0_224.html ) | [mobilenet-v2-1.0-224.py](scripts/mobilenet-v2-1.0-224.py) | 
|  mobilenet-v2-1.4-224   |  TensorFlow  |  1, 224, 224, 3  |  1245  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v2_1_4_224.html ) | [mobilenet-v2-1.4-224.py](scripts/mobilenet-v2-1.4-224.py) | 
|  mobilenet-v2-7  |  Onnx  |  1, 3, 224, 224  |  1224  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/mobilenet ) |  | 
|  mobilenet-v2-pytorch   |  PyTorch  |  1, 3, 224, 224  |  1315  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v2_pytorch.html ) | [mobilenet-v2-pytorch.py](scripts/mobilenet-v2-pytorch.py) | 
|  mobilenet-yolo-v4-syg   |  Keras  |  1, 416, 416, 3  |  738  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_yolo_v4_syg.html ) | [mobilenet-yolo-v4-syg.py](scripts/mobilenet-yolo-v4-syg.py) | 
|  mobilenetv2-12  |  Onnx  |  1, 3, 224, 224  |  1137  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx ) |  | 
|  nfnet-f0   |  PyTorch  |  1, 3, 256, 256  |  308  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_nfnet_f0.html ) | [nfnet-f0.py](scripts/nfnet-f0.py) | 
|  quantized_mobilenet_v2   |  Onnx  |  1, 224, 224, 3  |  1294  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mobilenetv2_quant.html ) | [quantized_mobilenet_v2.py](scripts/quantized_mobilenet_v2.py) | 
|  quantized_resnet18   |  Onnx  |  1, 224, 224, 3  |  945  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet_quant.html ) | [quantized_resnet18.py](scripts/quantized_resnet18.py) | 
|  quantized_resnet50   |  Onnx  |  1, 224, 224, 3  |  774  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet_quant.html ) | [quantized_resnet50.py](scripts/quantized_resnet50.py) | 
|  quantized_resnext101_32x8d   |  Onnx  |  1, 224, 224, 3  |  343  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext_quant.html ) | [quantized_resnext101_32x8d.py](scripts/quantized_resnext101_32x8d.py) | 
|  quantized_resnext101_64x4d   |  Onnx  |  1, 224, 224, 3  |  363  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext_quant.html ) | [quantized_resnext101_64x4d.py](scripts/quantized_resnext101_64x4d.py) | 
|  rcnn-ilsvrc13-9  |  Onnx  |  1, 3, 224, 224  |  534  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/rcnn_ilsvrc13/model ) |  | 
|  regnet_x_1_6gf   |  Onnx  |  1, 224, 224, 3  |  1042  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_1_6gf.py](scripts/regnet_x_1_6gf.py) | 
|  regnet_x_16gf   |  Onnx  |  1, 224, 224, 3  |  453  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_16gf.py](scripts/regnet_x_16gf.py) | 
|  regnet_x_3_2gf   |  Onnx  |  1, 224, 224, 3  |  865  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_3_2gf.py](scripts/regnet_x_3_2gf.py) | 
|  regnet_x_400mf   |  Onnx  |  1, 224, 224, 3  |  1255  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_400mf.py](scripts/regnet_x_400mf.py) | 
|  regnet_x_800mf   |  Onnx  |  1, 224, 224, 3  |  1228  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_800mf.py](scripts/regnet_x_800mf.py) | 
|  regnet_x_8gf   |  Onnx  |  1, 224, 224, 3  |  521  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_8gf.py](scripts/regnet_x_8gf.py) | 
|  regnet_y_1_6gf   |  Onnx  |  1, 224, 224, 3  |  696  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_1_6gf.py](scripts/regnet_y_1_6gf.py) | 
|  regnet_y_16gf   |  Onnx  |  1, 224, 224, 3  |  297  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_16gf.py](scripts/regnet_y_16gf.py) | 
|  regnet_y_3_2gf   |  Onnx  |  1, 224, 224, 3  |  612  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_3_2gf.py](scripts/regnet_y_3_2gf.py) | 
|  regnet_y_400mf   |  Onnx  |  1, 224, 224, 3  |  961  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_400mf.py](scripts/regnet_y_400mf.py) | 
|  regnet_y_800mf   |  Onnx  |  1, 224, 224, 3  |  980  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_800mf.py](scripts/regnet_y_800mf.py) | 
|  regnetx-3.2gf   |  PyTorch  |  1, 3, 224, 224  |  1785  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_regnetx_3_2gf.html ) | [regnetx-3.2gf.py](scripts/regnetx-3.2gf.py) | 
|  repvgg-a0   |  PyTorch  |  1, 3, 224, 224  |  947  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_a0.html ) | [repvgg-a0.py](scripts/repvgg-a0.py) | 
|  repvgg-b1   |  PyTorch  |  1, 3, 224, 224  |  451  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_b1.html ) | [repvgg-b1.py](scripts/repvgg-b1.py) | 
|  repvgg-b3   |  PyTorch  |  1, 3, 224, 224  |  272  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_b3.html ) | [repvgg-b3.py](scripts/repvgg-b3.py) | 
|  resnet-18-pytorch   |  PyTorch  |  1, 3, 224, 224  |  1128  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_18_pytorch.html ) | [resnet-18-pytorch.py](scripts/resnet-18-pytorch.py) | 
|  resnet-34-pytorch   |  PyTorch  |  1, 3, 224, 224  |  873  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_34_pytorch.html ) | [resnet-34-pytorch.py](scripts/resnet-34-pytorch.py) | 
|  resnet-50-pytorch   |  PyTorch  |  1, 3, 224, 224  |  802  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_50_pytorch.html ) | [resnet-50-pytorch.py](scripts/resnet-50-pytorch.py) | 
|  resnet-50-tf   |  TensorFlow  |  1, 224, 224, 3  |  805  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_50_tf.html ) | [resnet-50-tf.py](scripts/resnet-50-tf.py) | 
|  resnet101   |  PyTorch  |  1, 224, 224, 3  |  528  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html ) | [resnet101.py](scripts/resnet101.py) | 
|  resnet101   |  Onnx  |  1, 3, 224, 224  |  559  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet101.py](scripts/resnet101.py) | 
|  resnet101-v1-7  |  Onnx  |  1, 3, 224, 224  |  539  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet101-v1-7.onnx ) |  | 
|  resnet152   |  PyTorch  |  1, 224, 224, 3  |  421  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html ) | [resnet152.py](scripts/resnet152.py) | 
|  resnet152   |  Onnx  |  1, 3, 224, 224  |  453  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet152.py](scripts/resnet152.py) | 
|  resnet152-v1-7  |  Onnx  |  1, 3, 224, 224  |  433  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet152-v1-7.onnx ) |  | 
|  resnet18   |  PyTorch  |  1, 224, 224, 3  |  1114  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html ) | [resnet18.py](scripts/resnet18.py) | 
|  resnet18   |  Onnx  |  1, 3, 224, 224  |  1022  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet18.py](scripts/resnet18.py) | 
|  resnet34   |  PyTorch  |  1, 224, 224, 3  |  821  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html ) | [resnet34.py](scripts/resnet34.py) | 
|  resnet34   |  Onnx  |  1, 3, 224, 224  |  857  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet34.py](scripts/resnet34.py) | 
|  resnet50   |  Onnx  |  1, 3, 224, 224  |  762  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet50.py](scripts/resnet50.py) | 
|  resnet50-v1-12  |  Onnx  |  1, 3, 224, 224  |  730  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx ) |  | 
|  resnet50-v1-7  |  Onnx  |  1, 3, 224, 224  |  774  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/resnet/README.md ) |  | 
|  resnet50-v2-7  |  Onnx  |  1, 3, 224, 224  |  710  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/resnet/README.md ) |  | 
|  resnext101_32x8d   |  Onnx  |  1, 224, 224, 3  |  359  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext101_32x8d.py](scripts/resnext101_32x8d.py) | 
|  resnext101_64x4d   |  Onnx  |  1, 224, 224, 3  |  359  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext101_64x4d.py](scripts/resnext101_64x4d.py) | 
|  resnext50_32x4d   |  Onnx  |  1, 224, 224, 3  |  743  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext50_32x4d.py](scripts/resnext50_32x4d.py) | 
|  resnext50-32x4d   |  PyTorch  |  1, 3, 224, 224  |  696  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html ) |  | 
|  single-human-pose-estimation-0001   |  PyTorch  |  1, 3, 384, 288  |  230  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_single_human_pose_estimation_0001.html ) | [single-human-pose-estimation-0001.py](scripts/single-human-pose-estimation-0001.py) | 
|  squeezenet1_0   |  Onnx  |  1, 224, 224, 3  |  1446  |  [Torchvision Link]( https://pytorch.org/vision/main/models/squeezenet.html ) | [squeezenet1_0.py](scripts/squeezenet1_0.py) | 
|  squeezenet1_1   |  Onnx  |  1, 224, 224, 3  |  1573  |  [Torchvision Link]( https://pytorch.org/vision/main/models/squeezenet.html ) | [squeezenet1_1.py](scripts/squeezenet1_1.py) | 
|  squeezenet1.1-7  |  Onnx  |  1, 3, 224, 224  |  1494  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/squeezenet/model ) |  | 
|  vgg11   |  Onnx  |  1, 224, 224, 3  |  266  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg11.py](scripts/vgg11.py) | 
|  vgg11_bn   |  Onnx  |  1, 224, 224, 3  |  256  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg11_bn.py](scripts/vgg11_bn.py) | 
|  vgg13   |  Onnx  |  1, 224, 224, 3  |  259  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg13.py](scripts/vgg13.py) | 
|  vgg13_bn   |  Onnx  |  1, 224, 224, 3  |  258  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg13_bn.py](scripts/vgg13_bn.py) | 
|  vgg16   |  Onnx  |  1, 224, 224, 3  |  237  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg16.py](scripts/vgg16.py) | 
|  vgg16_bn   |  Onnx  |  1, 224, 224, 3  |  239  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg16_bn.py](scripts/vgg16_bn.py) | 
|  vgg16-bn-7  |  Onnx  |  1, 3, 224, 224  |  228  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-bn-7.onnx ) |  | 
|  vgg19   |  PyTorch  |  1, 224, 224, 3  |  221  |  [Torchvision Link]( https://pytorch.org/vision/0.12/generated/torchvision.models.vgg19.html ) | [vgg19.py](scripts/vgg19.py) | 
|  vgg19   |  Onnx  |  1, 3, 224, 224  |  223  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg19.py](scripts/vgg19.py) | 
|  vgg19_bn   |  Onnx  |  1, 224, 224, 3  |  223  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg19_bn.py](scripts/vgg19_bn.py) | 
|  vgg19-7  |  Onnx  |  1, 3, 224, 224  |  229  |  [ONNX Zoo Link]( https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg19-7.onnx ) |  | 
|  vgg19-bn   |  Onnx  |  1, 3, 224, 224  |  225  |  [Torchvision Link]( https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html ) |  | 
|  vgg19-bn-7  |  Onnx  |  1, 3, 224, 224  |  228  |  [ONNX Zoo Link]( https://github.com/onnx/models/tree/main/vision/classification/vgg/model ) |  | 
|  wide_resnet101_2   |  Onnx  |  1, 224, 224, 3  |  276  |  [Torchvision Link]( https://pytorch.org/vision/main/models/wide_resnet.html ) | [wide_resnet101_2.py](scripts/wide_resnet101_2.py) | 
|  wide_resnet50_2   |  Onnx  |  1, 224, 224, 3  |  438  |  [Torchvision Link]( https://pytorch.org/vision/main/models/wide_resnet.html ) | [wide_resnet50_2.py](scripts/wide_resnet50_2.py) | 
|  wide-resnet101-2   |  PyTorch  |  1, 3, 224, 224  |  262  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet101_2.html ) |  | 
|  wide-resnet50-2   |  PyTorch  |  1, 3, 224, 224  |  389  |  [Torchvision Link]( https://pytorch.org/vision/master/models/generated/torchvision.models.wide_resnet50_2.html ) |  | 
|  yolo-v2-tiny-tf   |  TensorFlow  |  1, 416, 416, 3  |  1090  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v2_tiny_tf.html ) | [yolo-v2-tiny-tf.py](scripts/yolo-v2-tiny-tf.py) | 
|  yolo-v3-tf   |  TensorFlow  |  1, 416, 416, 3  |  326  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v3_tf.html ) | [yolo-v3-tf.py](scripts/yolo-v3-tf.py) | 
|  yolo-v3-tiny-tf   |  TensorFlow  |  1, 416, 416, 3  |  1123  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v3_tiny_tf.html ) | [yolo-v3-tiny-tf.py](scripts/yolo-v3-tiny-tf.py) | 
|  yolo-v4-tf   |  TensorFlow  |  1, 416, 416, 3  |  195  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v4_tf.html ) | [yolo-v4-tf.py](scripts/yolo-v4-tf.py) | 
|  yolo-v4-tiny-tf   |  Keras  |  1, 416, 416, 3  |  966  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v4_tiny_tf.html ) | [yolo-v4-tiny-tf.py](scripts/yolo-v4-tiny-tf.py) | 
|  yolof   |  PyTorch  |  1, 3, 608, 608  |  109  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolof.html ) | [yolof.py](scripts/yolof.py) | 
|  Resnet18_AIMET   |  PyTorch  |  1, 3, 224, 224  |  1114  |  [Torchvision Link]( https://pytorch.org/vision/0.11/models.html#classification ) |  | 
|  Resnet50_AIMET   |  PyTorch  |  1, 3, 224, 224  |  774  |  [Torchvision Link]( https://pytorch.org/vision/0.11/models.html#classification ) |  | 
|  Resnet101_AIMET   |  PyTorch  |  1, 3, 224, 224  |  528  |  [Torchvision Link]( https://pytorch.org/vision/0.11/models.html#classification ) |  | 
|  Regnet_x_3_2gf_AIMET   |  PyTorch  |  1, 224, 224, 3  |  865  |  [Torchvision Link]( https://pytorch.org/vision/0.11/models.html#classification ) |  | 
|  ResNeXt101_AIMET   |  PyTorch  |  1, 224, 224, 3  |  351  |  [Torchvision Link]( https://pytorch.org/vision/0.11/models.html#classification ) |  | 
|  zfnet512-9  |  Onnx  |  1, 3, 224, 224  |  393  |  [ONNX Zoo Link]( https://github.com/onnx/onnx-tensorflow/wiki/ModelZoo-Status-(tag=v1.9.0)#24-zfnet-512 ) |  | 

## Torchvision ##
## ONNX Model Zoo ##
## OpenVINO ##

# Model Calibration/Compilation #

# Performance Estimation #


