![alt text](https://sima.ai/wp-content/uploads/2022/01/SiMaAI_Logo_Trademarked_Digital_FullColor_Large.svg)
# README #

Welcome to SiMa.ai's ML Model webpage!

A total of TBD models are supported on the SiMa.ai platform as part of [Palette V1.1 SDK](https://bit.ly/41q4tQT).

This Model List:

- Covers Multiple frameworks such as PyTorch and ONNX.
- Draws from various repositories including Torchvision, Open Model Zoo for OpenVINO, ONNX Model Zoo

For all TBD supported models, links/instructions are provided for the pre-trained FP32 models along with compilation scripts and PyTorch to ONNX conversion script.




# Model List #

|  Model   |  Framework  |  Input Shape  |  FPS  |  Pretrained Model |  Compilation script |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|  alexnet   |  Onnx  |  1, 224, 224, 3  |  1447  |  [Torchvision Link]( https://pytorch.org/vision/main/models/alexnet.html ) | [alexnet.py](scripts/alexnet.py) | 
|  ctdet_coco_dlav0_512   |  PyTorch  |  1, 3, 512, 512  |  996  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_ctdet_coco_dlav0_512.html ) | [ctdet_coco_dlav0_512.py](scripts/ctdet_coco_dlav0_512.py) | 
|  densenet121   |  PyTorch  |  1, 224, 224, 3  |  1461  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet121.py](scripts/densenet121.py) | 
|  densenet121   |  Onnx  |  1, 3, 224, 224  |  1434  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet121.py](scripts/densenet121.py) | 
|  densenet161   |  PyTorch  |  1, 224, 224, 3  |  903  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet161.py](scripts/densenet161.py) | 
|  densenet161   |  Onnx  |  1, 3, 224, 224  |  912  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet161.py](scripts/densenet161.py) | 
|  densenet169   |  PyTorch  |  1, 224, 224, 3  |  1080  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet169.py](scripts/densenet169.py) | 
|  densenet169   |  Onnx  |  1, 3, 224, 224  |  1122  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet169.py](scripts/densenet169.py) | 
|  densenet201   |  PyTorch  |  1, 224, 224, 3  |  831  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet201.py](scripts/densenet201.py) | 
|  densenet201   |  Onnx  |  1, 3, 224, 224  |  848  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet201.py](scripts/densenet201.py) | 
|  dla-34   |  PyTorch  |  1, 3, 224, 224  |  2769  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_dla_34.html ) | [dla-34.py](scripts/dla-34.py) | 
|  efficientnet_b0   |  Onnx  |  1, 224, 224, 3  |  2264  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b0.py](scripts/efficientnet_b0.py) | 
|  efficientnet_b1   |  Onnx  |  1, 224, 224, 3  |  1767  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b1.py](scripts/efficientnet_b1.py) | 
|  efficientnet_b2   |  Onnx  |  1, 224, 224, 3  |  1728  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b2.py](scripts/efficientnet_b2.py) | 
|  efficientnet_b3   |  Onnx  |  1, 224, 224, 3  |  1478  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b3.py](scripts/efficientnet_b3.py) | 
|  efficientnet_b4   |  Onnx  |  1, 224, 224, 3  |  1263  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b4.py](scripts/efficientnet_b4.py) | 
|  efficientnet_b5   |  Onnx  |  1, 224, 224, 3  |  949  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b5.py](scripts/efficientnet_b5.py) | 
|  efficientnet_b6   |  Onnx  |  1, 224, 224, 3  |  805  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b6.py](scripts/efficientnet_b6.py) | 
|  efficientnet_b7   |  Onnx  |  1, 224, 224, 3  |  622  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_b7.py](scripts/efficientnet_b7.py) | 
|  efficientnet_v2_m   |  Onnx  |  1, 224, 224, 3  |  727  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_v2_m.py](scripts/efficientnet_v2_m.py) | 
|  efficientnet_v2_s   |  Onnx  |  1, 224, 224, 3  |  978  |  [Torchvision Link]( https://pytorch.org/vision/main/models/efficientnet.html ) | [efficientnet_v2_s.py](scripts/efficientnet_v2_s.py) | 
|  efficientnet-b0-pytorch   |  PyTorch  |  1, 3, 224, 224  |  2084  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_b0_pytorch.html ) | [efficientnet-b0-pytorch.py](scripts/efficientnet-b0-pytorch.py) | 
|  efficientnet-v2-b0   |  PyTorch  |  1, 3, 224, 224  |  1935  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_v2_b0.html ) | [efficientnet-v2-b0.py](scripts/efficientnet-v2-b0.py) | 
|  efficientnet-v2-s   |  PyTorch  |  1, 224, 224, 3  |  953  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_efficientnet_v2_s.html ) | [efficientnet-v2-s.py](scripts/efficientnet-v2-s.py) | 
|  erfnet   |  PyTorch  |  1, 3, 208, 976  |  1558  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_erfnet.html ) | [erfnet.py](scripts/erfnet.py) | 
|  higher-hrnet-w32-human-pose-estimation   |  PyTorch  |  1, 3, 512, 512  |  500  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_higher_hrnet_w32_human_pose_estimation.html ) | [higher-hrnet-w32-human-pose-estimation.py](scripts/higher-hrnet-w32-human-pose-estimation.py) | 
|  human-pose-estimation-3d-0001   |  PyTorch  |  1, 3, 256, 448  |  1777  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_human_pose_estimation_3d_0001.html ) | [human-pose-estimation-3d-0001.py](scripts/human-pose-estimation-3d-0001.py) | 
|  mnasnet0_5   |  Onnx  |  1, 224, 224, 3  |  4029  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet0_5.py](scripts/mnasnet0_5.py) | 
|  mnasnet0_75   |  Onnx  |  1, 224, 224, 3  |  3392  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet0_75.py](scripts/mnasnet0_75.py) | 
|  mnasnet1_0   |  Onnx  |  1, 224, 224, 3  |  3767  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet1_0.py](scripts/mnasnet1_0.py) | 
|  mnasnet1_3   |  Onnx  |  1, 224, 224, 3  |  3518  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mnasnet.html ) | [mnasnet1_3.py](scripts/mnasnet1_3.py) | 
|  mobilenet_v2   |  Onnx  |  1, 224, 224, 3  |  3813  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mobilenetv2.html ) | [mobilenet_v2.py](scripts/mobilenet_v2.py) | 
|  mobilenet-v2-pytorch   |  PyTorch  |  1, 3, 224, 224  |  3914  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v2_pytorch.html ) | [mobilenet-v2-pytorch.py](scripts/mobilenet-v2-pytorch.py) | 
|  mobilenet-yolo-v4-syg   |  Keras  |  1, 416, 416, 3  |  2199  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_yolo_v4_syg.html ) | [mobilenet-yolo-v4-syg.py](scripts/mobilenet-yolo-v4-syg.py) | 
|  nfnet-f0   |  PyTorch  |  1, 3, 256, 256  |  922  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_nfnet_f0.html ) | [nfnet-f0.py](scripts/nfnet-f0.py) | 
|  quantized_mobilenet_v2   |  Onnx  |  1, 224, 224, 3  |  3851  |  [Torchvision Link]( https://pytorch.org/vision/main/models/mobilenetv2_quant.html ) | [quantized_mobilenet_v2.py](scripts/quantized_mobilenet_v2.py) | 
|  quantized_resnet18   |  Onnx  |  1, 224, 224, 3  |  2892  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet_quant.html ) | [quantized_resnet18.py](scripts/quantized_resnet18.py) | 
|  quantized_resnet50   |  Onnx  |  1, 224, 224, 3  |  2304  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet_quant.html ) | [quantized_resnet50.py](scripts/quantized_resnet50.py) | 
|  quantized_resnext101_32x8d   |  Onnx  |  1, 224, 224, 3  |  1023  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext_quant.html ) | [quantized_resnext101_32x8d.py](scripts/quantized_resnext101_32x8d.py) | 
|  quantized_resnext101_64x4d   |  Onnx  |  1, 224, 224, 3  |  1091  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext_quant.html ) | [quantized_resnext101_64x4d.py](scripts/quantized_resnext101_64x4d.py) | 
|  regnet_x_1_6gf   |  Onnx  |  1, 224, 224, 3  |  3074  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_1_6gf.py](scripts/regnet_x_1_6gf.py) | 
|  regnet_x_16gf   |  Onnx  |  1, 224, 224, 3  |  1358  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_16gf.py](scripts/regnet_x_16gf.py) | 
|  regnet_x_3_2gf   |  Onnx  |  1, 224, 224, 3  |  2527  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_3_2gf.py](scripts/regnet_x_3_2gf.py) | 
|  regnet_x_400mf   |  Onnx  |  1, 224, 224, 3  |  3733  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_400mf.py](scripts/regnet_x_400mf.py) | 
|  regnet_x_800mf   |  Onnx  |  1, 224, 224, 3  |  3647  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_800mf.py](scripts/regnet_x_800mf.py) | 
|  regnet_x_8gf   |  Onnx  |  1, 224, 224, 3  |  1555  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_x_8gf.py](scripts/regnet_x_8gf.py) | 
|  regnet_y_1_6gf   |  Onnx  |  1, 224, 224, 3  |  2078  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_1_6gf.py](scripts/regnet_y_1_6gf.py) | 
|  regnet_y_16gf   |  Onnx  |  1, 224, 224, 3  |  891  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_16gf.py](scripts/regnet_y_16gf.py) | 
|  regnet_y_3_2gf   |  Onnx  |  1, 224, 224, 3  |  1821  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_3_2gf.py](scripts/regnet_y_3_2gf.py) | 
|  regnet_y_400mf   |  Onnx  |  1, 224, 224, 3  |  2832  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_400mf.py](scripts/regnet_y_400mf.py) | 
|  regnet_y_800mf   |  Onnx  |  1, 224, 224, 3  |  2925  |  [Torchvision Link]( https://pytorch.org/vision/main/models/regnet.html ) | [regnet_y_800mf.py](scripts/regnet_y_800mf.py) | 
|  repvgg-a0   |  PyTorch  |  1, 3, 224, 224  |  2818  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_a0.html ) | [repvgg-a0.py](scripts/repvgg-a0.py) | 
|  repvgg-b1   |  PyTorch  |  1, 3, 224, 224  |  1350  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_b1.html ) | [repvgg-b1.py](scripts/repvgg-b1.py) | 
|  repvgg-b3   |  PyTorch  |  1, 3, 224, 224  |  815  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_repvgg_b3.html ) | [repvgg-b3.py](scripts/repvgg-b3.py) | 
|  resnet-18-pytorch   |  PyTorch  |  1, 3, 224, 224  |  3401  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_18_pytorch.html ) | [resnet-18-pytorch.py](scripts/resnet-18-pytorch.py) | 
|  resnet-34-pytorch   |  PyTorch  |  1, 3, 224, 224  |  2593  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_34_pytorch.html ) | [resnet-34-pytorch.py](scripts/resnet-34-pytorch.py) | 
|  resnet-50-pytorch   |  PyTorch  |  1, 3, 224, 224  |  2385  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_50_pytorch.html ) | [resnet-50-pytorch.py](scripts/resnet-50-pytorch.py) | 
|  resnet-50-tf   |  TensorFlow  |  1, 224, 224, 3  |  2396  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_resnet_50_tf.html ) | [resnet-50-tf.py](scripts/resnet-50-tf.py) | 
|  resnet101   |  PyTorch  |  1, 224, 224, 3  |  1596  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html ) | [resnet101.py](scripts/resnet101.py) | 
|  resnet101   |  Onnx  |  1, 3, 224, 224  |  1667  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet101.py](scripts/resnet101.py) | 
|  resnet152   |  PyTorch  |  1, 224, 224, 3  |  1263  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html ) | [resnet152.py](scripts/resnet152.py) | 
|  resnet152   |  Onnx  |  1, 3, 224, 224  |  1356  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet152.py](scripts/resnet152.py) | 
|  resnet18   |  PyTorch  |  1, 224, 224, 3  |  3309  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html ) | [resnet18.py](scripts/resnet18.py) | 
|  resnet18   |  Onnx  |  1, 3, 224, 224  |  3136  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet18.py](scripts/resnet18.py) | 
|  resnet34   |  PyTorch  |  1, 224, 224, 3  |  2427  |  [Torchvision Link]( https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html ) | [resnet34.py](scripts/resnet34.py) | 
|  resnet34   |  Onnx  |  1, 3, 224, 224  |  2538  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet34.py](scripts/resnet34.py) | 
|  resnet50   |  Onnx  |  1, 3, 224, 224  |  2251  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet50.py](scripts/resnet50.py) | 
|  resnext101_32x8d   |  Onnx  |  1, 224, 224, 3  |  1072  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext101_32x8d.py](scripts/resnext101_32x8d.py) | 
|  resnext101_64x4d   |  Onnx  |  1, 224, 224, 3  |  1079  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext101_64x4d.py](scripts/resnext101_64x4d.py) | 
|  resnext50_32x4d   |  Onnx  |  1, 224, 224, 3  |  2223  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnext.html ) | [resnext50_32x4d.py](scripts/resnext50_32x4d.py) | 
|  single-human-pose-estimation-0001   |  PyTorch  |  1, 3, 384, 288  |  689  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_single_human_pose_estimation_0001.html ) | [single-human-pose-estimation-0001.py](scripts/single-human-pose-estimation-0001.py) | 
|  squeezenet1_0   |  Onnx  |  1, 224, 224, 3  |  4292  |  [Torchvision Link]( https://pytorch.org/vision/main/models/squeezenet.html ) | [squeezenet1_0.py](scripts/squeezenet1_0.py) | 
|  squeezenet1_1   |  Onnx  |  1, 224, 224, 3  |  4661  |  [Torchvision Link]( https://pytorch.org/vision/main/models/squeezenet.html ) | [squeezenet1_1.py](scripts/squeezenet1_1.py) | 
|  vgg11   |  Onnx  |  1, 224, 224, 3  |  794  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg11.py](scripts/vgg11.py) | 
|  vgg11_bn   |  Onnx  |  1, 224, 224, 3  |  771  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg11_bn.py](scripts/vgg11_bn.py) | 
|  vgg13   |  Onnx  |  1, 224, 224, 3  |  773  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg13.py](scripts/vgg13.py) | 
|  vgg13_bn   |  Onnx  |  1, 224, 224, 3  |  771  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg13_bn.py](scripts/vgg13_bn.py) | 
|  vgg16   |  Onnx  |  1, 224, 224, 3  |  711  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg16.py](scripts/vgg16.py) | 
|  vgg16_bn   |  Onnx  |  1, 224, 224, 3  |  715  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg16_bn.py](scripts/vgg16_bn.py) | 
|  vgg19   |  PyTorch  |  1, 224, 224, 3  |  662  |  [Torchvision Link]( https://pytorch.org/vision/0.12/generated/torchvision.models.vgg19.html ) | [vgg19.py](scripts/vgg19.py) | 
|  vgg19   |  Onnx  |  1, 3, 224, 224  |  669  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg19.py](scripts/vgg19.py) | 
|  vgg19_bn   |  Onnx  |  1, 224, 224, 3  |  668  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg19_bn.py](scripts/vgg19_bn.py) | 
|  wide_resnet101_2   |  Onnx  |  1, 224, 224, 3  |  828  |  [Torchvision Link]( https://pytorch.org/vision/main/models/wide_resnet.html ) | [wide_resnet101_2.py](scripts/wide_resnet101_2.py) | 
|  wide_resnet50_2   |  Onnx  |  1, 224, 224, 3  |  1312  |  [Torchvision Link]( https://pytorch.org/vision/main/models/wide_resnet.html ) | [wide_resnet50_2.py](scripts/wide_resnet50_2.py) | 
|  yolo-v2-tiny-tf   |  TensorFlow  |  1, 416, 416, 3  |  3266  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v2_tiny_tf.html ) | [yolo-v2-tiny-tf.py](scripts/yolo-v2-tiny-tf.py) | 
|  yolo-v3-tf   |  TensorFlow  |  1, 416, 416, 3  |  977  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v3_tf.html ) | [yolo-v3-tf.py](scripts/yolo-v3-tf.py) | 
|  yolo-v3-tiny-tf   |  TensorFlow  |  1, 416, 416, 3  |  3328  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v3_tiny_tf.html ) | [yolo-v3-tiny-tf.py](scripts/yolo-v3-tiny-tf.py) | 
|  yolo-v4-tf   |  TensorFlow  |  1, 416, 416, 3  |  584  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v4_tf.html ) | [yolo-v4-tf.py](scripts/yolo-v4-tf.py) | 
|  yolo-v4-tiny-tf   |  Keras  |  1, 416, 416, 3  |  2922  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolo_v4_tiny_tf.html ) | [yolo-v4-tiny-tf.py](scripts/yolo-v4-tiny-tf.py) | 
|  yolof   |  PyTorch  |  1, 3, 608, 608  |  326  |  [OpenVINO Link]( https://docs.openvino.ai/2023.0/omz_models_model_yolof.html ) | [yolof.py](scripts/yolof.py) | 
# Palette Setup #

Installation of Palette software with command-line interface (CLI) option is required for all the steps listed below. It can be downloaded from [SiMa.ai website](https://developer.sima.ai/login?step=signIn&_gl=1*1a8w2o4*_ga*MTc1ODQwODQxMS4xNzAxOTAwMTM3*_ga_BRMYJE3DCD*MTcwMjI2MTYxOC4zLjAuMTcwMjI2MTYxOC42MC4wLjA.).

After downloading the Palette CLI SDK zip file, it can be unarchived and CLI docker container can be installed using steps below (below steps are verified on Linux Ubuntu system).

```
vikas_paliwal@instance-5:~/Downloads$ unzip SiMa_CLI_1.1.0_master_B40.zip 
Archive:  SiMa_CLI_1.1.0_master_B40.zip
   creating: 1.1.0_master_B40/
   creating: 1.1.0_master_B40/sima-cli/
  inflating: 1.1.0_master_B40/sima-cli/start.sh  
  inflating: 1.1.0_master_B40/sima-cli/simaclisdk_1_1_0_master_B40.tar  
  inflating: 1.1.0_master_B40/sima-cli/uninstall.sh  
  inflating: 1.1.0_master_B40/sima-cli/install.sh  
  inflating: 1.1.0_master_B40/sima-cli/release-notes.txt  
  inflating: 1.1.0_master_B40/sima-cli/install_dependencies.sh  
  inflating: 1.1.0_master_B40/sima-cli/stop.sh 
```

The Palette CLI unarchived folder is now available and its contents can be viewed using `ls`. Below steps were done for Palette 1.1 release so, for this release, the folder `1.1.0_master_B5` or something similar must be present. Enter the `1.1.0_master_B5` folder and the `sima-cli` folder in it. Here on listing the contents, scripts for installation (`install.sh`), starting the container (`start.sh`) and stopping the container (`stop.sh`) scripts should be visible. To kickstart the install procedure, issue the `install.sh` command and command line outputs must look similar to below.

```

vikas_paliwal@instance-5:~/Downloads$ cd 1.1.0_master_B40
vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40$ ls
sima-cli
vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40$ cd sima-cli/
vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40/sima-cli$ ls
install.sh  install_dependencies.sh  release-notes.txt  simaclisdk_1_1_0_master_B40.tar  start.sh  stop.sh  uninstall.sh
vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40/sima-cli$ ./install.sh
Checking if SiMa CLI version 1.1.0_master_B40 is already installed...
./install.sh: line 28: netstat: command not found
Enter work directory [/home/vikas_paliwal/workspace]: 
Loading Docker image version 1.1.0_master_B40...

d3fa9d362c05: Loading layer [==================================================>]  75.18MB/75.18MB
49e2acfe9ae4: Loading layer [==================================================>]  1.456GB/1.456GB
562015b533e7: Loading layer [==================================================>]   2.89MB/2.89MB
5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
1324083fec73: Loading layer [==================================================>]  3.363MB/3.363MB
8e5cc2fcb1e8: Loading layer [==================================================>]  3.355GB/3.355GB
43a255d427ea: Loading layer [==================================================>]  751.6kB/751.6kB
5757c7e05d4a: Loading layer [==================================================>]  52.34MB/52.34MB
8d4b34bc4b54: Loading layer [==================================================>]  569.9kB/569.9kB
89f40f6da13d: Loading layer [==================================================>]  103.8MB/103.8MB
debfb171ceab: Loading layer [==================================================>]  47.57MB/47.57MB
e764414c5d6f: Loading layer [==================================================>]  103.8MB/103.8MB
4bfc40f38fbb: Loading layer [==================================================>]  103.8MB/103.8MB
c6152ea993db: Loading layer [==================================================>]   2.56kB/2.56kB
06db2209b9b4: Loading layer [==================================================>]   5.12kB/5.12kB
c3342c240117: Loading layer [==================================================>]   7.68kB/7.68kB
7612cea1d7b4: Loading layer [==================================================>]  9.728kB/9.728kB
4a1cbab15e81: Loading layer [==================================================>]  24.58kB/24.58kB
3b62e012d34d: Loading layer [==================================================>]   2.56kB/2.56kB
24426ea09369: Loading layer [==================================================>]  8.596GB/8.596GB
08ecd0efb4ab: Loading layer [==================================================>]  3.584kB/3.584kB
f91c5ae2d614: Loading layer [==================================================>]  144.2MB/144.2MB
27397f5dacc7: Loading layer [==================================================>]  1.362GB/1.362GB
b050726a85f0: Loading layer [==================================================>]  110.6kB/110.6kB
5e23613dd9e4: Loading layer [==================================================>]  3.584kB/3.584kB
5ee25ce4026c: Loading layer [==================================================>]  4.608kB/4.608kB
Loaded image: simaclisdk:1.1.0_master_B40
Checking SiMa SDK Bridge Network...
SiMa SDK Bridge Network found.
9bb247385914543099e3aa5f601e29b2b8b0d9e65dfa3f6d985bbdb568acb378
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/1.1.0_master_B40/sima-cli/passwd.txt
Successfully copied 3.07kB to simaclisdk_1_1_0_master_B40:/etc/passwd
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/1.1.0_master_B40/sima-cli/shadow.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B40:/etc/shadow
Successfully copied 2.05kB to /home/vikas_paliwal/Downloads/1.1.0_master_B40/sima-cli/group.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B40:/etc/group
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/1.1.0_master_B40/sima-cli/sudoers.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B40:/etc/sudoers
Installation successful. To log in to the Docker container, please use the './start.sh' script
Your local work directory '/home/vikas_paliwal/workspace' has been mounted to '/home/docker/sima-cli'
```

At this point, the Palette CLI container is successfully installed. The container can now be started anytime using `start.sh` command and then all CLI commands can be run from inside the Palette CLI container as shown below.

```

vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40/sima-cli$ ./start.sh
Checking if the container is already running...
 ==> Container is already running. Proceeding to start an interactive shell.
vikas_paliwal@9bb247385914:/home$ 


```

# Downloading the Models #

SiMa.ai subset of compatible models references repositories like Torchvison,
ONNX model zoo and OpenVINO. These repositories offer pretrained models
in floating-point 32-bit (FP32) format that need to be quantized and compiled
for SiMa.ai&#39;s MLSoC using the Palette ModelSDK. To this end, certain helper
scripts are provided as part of this repository that fetch the models from original
pretrained model repositories. Here we describe instructions for getting the
models from these repositories. To review model details, refer to the original
papers, datasets, etc. mentioned in the corresponding source links provided.
Bring your data and get started on running models of interest on SiMa.ai&#39;s
MLSoC.
## Torchvision ##

[Torchvision](https://pytorch.org/vision/stable/models.html)'s `torchvision.models`subpackage offers ML model architectures
along with pretrained weights. SiMa.ai's ModelSDK can consume models from
PyTorch that include the model topology and weights: either using TorchScript,
or exporting the models to ONNX. Given developers familiarity with ONNX, this
repository provides a helper script ([torchvision_to_onnx.py](torchvision_to_onnx.py)) to download
the Torchvision model(s) and convert them to ONNX automatically.

- To use the script, either clone the repository or download and copy to the Palette CLI docker image.
```
vikas_paliwal@instance-5:~/Downloads/1.1.0_master_B40/sima-cli$ docker cp  ~/Downloads/torchvision_to_onnx.py 9bb247385914:/home
Successfully copied 3.58kB to 9bb247385914:/home
```

- From inside the Palette CLI container, the following command can be used to download and convert models:
```

vikas_paliwal@9bb247385914:/home$ sudo python3 torchvision_to_onnx.py --model_name densenet121
Downloading: "https://github.com/pytorch/vision/zipball/v0.16.0" to /root/.cache/torch/hub/v0.16.0.zip
/usr/local/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=DenseNet121_Weights.IMAGENET1K_V1`. You can also use `weights=DenseNet121_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
100%|��������������������������������������������������������������������������������������������������������| 30.8M/30.8M [00:00<00:00, 52.6MB/s]
Before torch.onnx.export tensor([[[[ 1.7745,  0.7670, -0.2136,  ..., -1.5743, -0.4873,  1.0913],
          [ 0.0137, -0.9518,  0.8827,  ..., -0.1733, -0.1817,  2.1811],
          [ 0.6135, -0.9099, -2.0007,  ...,  0.3961, -0.4789, -1.5344],
          ...,
          [-1.1500, -0.1356,  0.5894,  ..., -1.2137,  0.8792,  0.6761],
          [-0.3458, -0.6029,  0.9585,  ...,  0.0141, -1.8495, -0.9339],
          [-0.4006, -1.1134, -0.3972,  ..., -0.5107, -0.8084, -1.4360]]]])
============== Diagnostic Run torch.onnx.export version 2.0.1+cpu ==============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

After torch.onnx.export

```
- The downloaded and converted model can be viewed as below.
```
vikas_paliwal@9bb247385914:/home$ ls
densenet121.onnx  docker  torchvision_to_onnx.py  

```

- The model is now successully downloaded from Torchvision repository and ready for usage with Palette CLI tools.


# Model Calibration/Compilation #

Helper scripts to compile each model are provided through this repository. The
source code for these helper scripts can be reviewed using links in the Model
List section. These compiler scripts come with multiple preconfigured settings
for input resolutions, calibration scheme, quantization method etc. These can
be adjusted per needs and full details on how to exercise various compile
options are provided in Palette CLI User Guide, available through SiMa.ai
developer zone. After cloning this repository, the user should download the
model of interest, and access the corresponding script for that model. It is
important to ensure the path of model file in the helper script, referenced
through model_path variable, is correct. 

- The model can be compiled from the Palette docker using this helper script with the command:`python3 [HELPER_SCRIPT]`
```
vikas_paliwal@9bb247385914:/home$ sudo python3 scripts/densenet121/densenet121.py
Model SDK version: 1.1.0

Running calibration ...DONE
2023-12-14 17:51:47,636 - afe.ir.quantization_utils - WARNING - Quantized bias was clipped, resulting in precision loss.  Model may need retraining.
2023-12-14 17:51:49,554 - afe.ir.quantization_utils - WARNING - Quantized bias was clipped, resulting in precision loss.  Model may need retraining.
Running quantization ...DONE


```

- After successful compilation, the resulting files are generated in `result/[MODEL_NAME_CALIBRATION_OPTIONS]/mpk` folder which now has `*.yaml, *.json, *.lm` generated as outputs of compilation. These files together can be used for performance estimation as described in next section.
```
vikas_paliwal@9bb247385914:/home$ ls
debug.log  docker  models  models.zip  result  scripts	torchvision_to_onnx.py	vikas_paliwal
vikas_paliwal@9bb247385914:/home$ ls result/
densenet121_asym_True_per_ch_True
vikas_paliwal@9bb247385914:/home$ ls result/densenet121_asym_True_per_ch_True/
mpk
vikas_paliwal@9bb247385914:/home$ ls result/densenet121_asym_True_per_ch_True/mpk
densenet121_mpk.tar.gz	densenet121_stage1_mla_compressed.mlc  densenet121_stage1_mla.ifm.mlc  densenet121_stage1_mla.mlc  densenet121_stage1_mla.ofm_chk.mlc

```


# License #
The primary license for the models in the [SiMa.ai Model Zoo](https://github.com/SiMa-ai/models) is the BSD 3-Clause License, see [LICENSE](LICENSE.txt). However:



Certain models may be subject to additional restrictions and/or terms. To the extent a LICENSE.txt file is provided for a particular model, please review the LICENSE.txt file carefully as you are responsible for your compliance with such restrictions and/or terms.

