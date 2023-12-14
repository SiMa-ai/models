![alt text](https://sima.ai/wp-content/uploads/2022/01/SiMaAI_Logo_Trademarked_Digital_FullColor_Large.svg)
# README #

Welcome to SiMa.ai's ML Model webpage!

A total of TBD models are supported on the SiMa.ai platform as part of [Palette V1.1 SDK](https://bit.ly/41q4tQT).

This Model List:

- Covers Multiple frameworks such as PyTorch and ONNX.
- Draws from various repositories including Torchvision, Open Model Zoo for OpenVINO, ONNX Model Zoo

For all TBD supported models, links/instructions are provided for the pre-trained FP32 models along with compilation scripts and PyTorch to ONNX conversion script.




# Model List #

|  Model   |  Framework  |  Input Shape  |  Pretrained Model |  Compilation script |
| ------------- | ------------- | ------------- | ------------- | ------------- |
|  densenet121   |  PyTorch  |  1, 224, 224, 3  |  [Torchvision Link]( https://pytorch.org/vision/main/models/densenet.html ) | [densenet121.py](scripts/densenet121.py) | 
|  resnet50   |  Onnx  |  1, 3, 224, 224  |  [Torchvision Link]( https://pytorch.org/vision/main/models/resnet.html ) | [resnet50.py](scripts/resnet50.py) | 
|  vgg16   |  Onnx  |  1, 224, 224, 3  |  [Torchvision Link]( https://pytorch.org/vision/main/models/vgg.html ) | [vgg16.py](scripts/vgg16.py) | 
|  vgg19   |  PyTorch  |  1, 224, 224, 3  |  [Torchvision Link]( https://pytorch.org/vision/0.12/generated/torchvision.models.vgg19.html ) | [vgg19.py](scripts/vgg19.py) | 

# Palette Setup #

Installation of Palette software with command-line interface (CLI) option is required for all the steps listed below. It can be downloaded from [SiMa.ai website](https://developer.sima.ai/login?step=signIn&_gl=1*1a8w2o4*_ga*MTc1ODQwODQxMS4xNzAxOTAwMTM3*_ga_BRMYJE3DCD*MTcwMjI2MTYxOC4zLjAuMTcwMjI2MTYxOC42MC4wLjA.).

After downloading the Palette CLI SDK zip file, it can be unarchived and CLI docker container can be installed using steps below (below steps are verified on Linux Ubuntu system).

```



```

The Palette CLI unarchived folder is now available and its contents can be viewed using `ls`. Below steps were done for Palette 1.1 release so, for this release, the folder `1.1.0_master_B5` or something similar must be present. Enter the `1.1.0_master_B5` folder and the `sima-cli` folder in it. Here on listing the contents, scripts for installation (`install.sh`), starting the container (`start.sh`) and stopping the container (`stop.sh`) scripts should be visible. To kickstart the install procedure, issue the `install.sh` command and command line outputs must look similar to below.

```

vikas_paliwal@instance-5:~/Downloads/palette$ ls
1.1.0_master_B5  SiMa_CLI_1.1.0_master_B5.zip
vikas_paliwal@instance-5:~/Downloads/palette$ cd 1.1.0_master_B5/
vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5$ ls
sima-cli
vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5$ cd sima-cli/
vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5/sima-cli$ ls
install.sh  install_dependencies.sh  release-notes.txt  simaclisdk_1_1_0_master_B5.tar  start.sh  stop.sh  uninstall.sh
vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5/sima-cli$ ./install.sh 
Docker daemon is not running.
If you have Docker installed, make sure it is running and try again.
vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5/sima-cli$ sudo ./install.sh 
Checking if SiMa CLI version 1.1.0_master_B5 is already installed...
Do you want to reinstall it? [y/N] y
./install.sh: line 28: netstat: command not found
Enter work directory [/home/vikas_paliwal/workspace]: 
Loading Docker image version 1.1.0_master_B5...
6c3e7df31590: Loading layer [==================================================>]  75.17MB/75.17MB
d0757cff9a50: Loading layer [==================================================>]  1.456GB/1.456GB
d363a5d9797a: Loading layer [==================================================>]   2.89MB/2.89MB
5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
1192b82b51e1: Loading layer [==================================================>]  245.3MB/245.3MB
f49434ad8f02: Loading layer [==================================================>]  3.363MB/3.363MB
7ad3f4bc4344: Loading layer [==================================================>]  3.364GB/3.364GB
e30ef4fd0a5e: Loading layer [==================================================>]  714.8kB/714.8kB
61d4e7a65eb0: Loading layer [==================================================>]  52.22MB/52.22MB
87e980650c89: Loading layer [==================================================>]  547.8kB/547.8kB
5e1c8475cb15: Loading layer [==================================================>]  103.8MB/103.8MB
f2c4ca8dd483: Loading layer [==================================================>]  47.57MB/47.57MB
fea17013a693: Loading layer [==================================================>]  103.8MB/103.8MB
ac1a9ed8be83: Loading layer [==================================================>]  103.8MB/103.8MB
1be0dedb722a: Loading layer [==================================================>]   2.56kB/2.56kB
60cb090989cb: Loading layer [==================================================>]   5.12kB/5.12kB
fe2946c9c0cb: Loading layer [==================================================>]   7.68kB/7.68kB
7b1ae9dd2fda: Loading layer [==================================================>]  9.728kB/9.728kB
53dd9ecce6a2: Loading layer [==================================================>]  24.58kB/24.58kB
7054a48d829c: Loading layer [==================================================>]   2.56kB/2.56kB
f17c57fd7d3c: Loading layer [==================================================>]  8.565GB/8.565GB
b2681143c79e: Loading layer [==================================================>]  73.73kB/73.73kB
7c24547aa974: Loading layer [==================================================>]  144.2MB/144.2MB
a6d76c5f0c1b: Loading layer [==================================================>]  1.408GB/1.408GB
392ea411a290: Loading layer [==================================================>]  101.4kB/101.4kB
e6c0f8f854be: Loading layer [==================================================>]  3.584kB/3.584kB
c2124b1e3113: Loading layer [==================================================>]  4.608kB/4.608kB
Loaded image: simaclisdk:1.1.0_master_B5
Checking SiMa SDK Bridge Network...
SiMa SDK Bridge Network found.
59d339853bd13cb7f69911fc5689d272d8dec8407f316417ca4b59f82b3fb884
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/palette/1.1.0_master_B5/sima-cli/passwd.txt
Successfully copied 3.07kB to simaclisdk_1_1_0_master_B5:/etc/passwd
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/palette/1.1.0_master_B5/sima-cli/shadow.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B5:/etc/shadow
Successfully copied 2.05kB to /home/vikas_paliwal/Downloads/palette/1.1.0_master_B5/sima-cli/group.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B5:/etc/group
Successfully copied 2.56kB to /home/vikas_paliwal/Downloads/palette/1.1.0_master_B5/sima-cli/sudoers.txt
Successfully copied 2.56kB to simaclisdk_1_1_0_master_B5:/etc/sudoers
Installation successful. To log in to the Docker container, please use the './start.sh' script
Your local work directory '/home/vikas_paliwal/workspace' has been mounted to '/home/docker/sima-cli'

```

At this point, the Palette CLI container is successfully installed. The container can now be started anytime using `start.sh` command and then all CLI commands can be run from inside the Palette CLI container as shown below.

```

vikas_paliwal@instance-5:~/Downloads/palette/1.1.0_master_B5/sima-cli$ sudo ./start.sh
Checking if the container is already running...
 ==> Container is already running. Proceeding to start an interactive shell.
root@59d339853bd1:/home# 


```

# Downloading the Models #

SiMa.ai model zoo currently offers sample models from repositories like Torchvison, ONNX model zoo and OpenVINO. These repositories offer pretrained models in floating-point 32-bit (FP32) format that eventually needs to be quantized and compiled for SiMa.ai's MLSoC. To this end, certain helper scripts are provided as part of this repository that fetch the models from original pretrained model repositories and optionally convert them to a format readily accepted by Palette CLI tool. Here we describe instructions for getting the models from these repositories.

It must be mentioned that users of Palette CLI may need to review the model details, original research papers, datasets used etc. For these purposes, links to the pretrained model repositories for each of the available sample models is provided. Users are encouraged to review the information available on these repositories to gain a better understandin of model details. 

## Torchvision ##

[Torchvision](https://pytorch.org/vision/stable/models.html)'s `torchvision.models` subpackage offers ML model architectures along with pretrained weights. This repository provides helper Python script [`torchvision_to_onnx.py`](torchvision_to_onnx.py) to download the Torchvision model(s). 

Download the [torchvision_to_onnx.py](torchvision_to_onnx.py) to the local system and, from the folder containing this file, issue below command to move the file inside container using the standard construct of [docker copy command](https://docs.docker.com/engine/reference/commandline/cp/).

```

vikas_paliwal@instance-5:~/Downloads/sima-ai-qa-5cacd287b0ad/sima-cli/libs$ sudo docker cp  torchvision_to_onnx.py 59d339853bd1:/home
Successfully copied 3.58kB to 59d339853bd1:/home

```

From inside the Palette CLI container, this command can be used to fetch and convert a model from list above that comes from Torchvision. E.g. to download a model like `densenet121`, the script can be used as shown below. Upon listing the folder content using `ls`, the newly downloaded pretrained model `resnet101.onnx` must be visible, as below.

```

root@59d339853bd1:/home# python3 torchvision_to_onnx.py --model-name densenet121
usage: torchvision_to_onnx.py [-h] --model_name MODEL_NAME
torchvision_to_onnx.py: error: the following arguments are required: --model_name
root@59d339853bd1:/home# python3 torchvision_to_onnx.py --model_name densenet121
Using cache found in /root/.cache/torch/hub/pytorch_vision_v0.16.0
/usr/local/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=DenseNet121_Weights.IMAGENET1K_V1`. You can also use `weights=DenseNet121_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Before torch.onnx.export tensor([[[[ 0.2556, -1.1689, -0.9834,  ..., -0.6764, -0.7731, -0.1609],
          [-0.4004,  0.3737, -0.1782,  ..., -0.0773,  0.8424, -1.0558],
          [-0.0491, -0.0103, -0.6932,  ...,  1.8154,  0.4333,  1.4905],
          ...,
          [-0.4963,  1.3916, -0.4413,  ..., -1.1120,  0.0762,  0.8303],
          [ 1.5159, -0.8136, -0.3001,  ...,  0.2860,  0.6737, -1.0139],
          [-0.2186,  0.5211, -0.3220,  ..., -0.3507,  0.6220, -0.4348]]]])
============== Diagnostic Run torch.onnx.export version 2.0.1+cpu ==============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

After torch.onnx.export

root@59d339853bd1:/home# ls
debug.log  densenet121.onnx  docker  result  root  torchvision_to_onnx.py

```

The model is now successully downloaded from Torchvision repository and ready for usage with Palette CLI tools.


# Model Calibration/Compilation #
Helper script to compile the models are provided for each ML model offered through this repository. The source code for these helper scripts can be reviewed using links provided in above **Model List** section. These compiler scipts come with multiple preconfigured settings for input resolutions, calibration scheme, quantization method etc. These can be adjusted per needs and full details on how to exercise various compile options are provided in Palette CLI User Guide, available through [SiMa.ai developer zone](https://developer.sima.ai/login?step=signIn&_gl=1*1a8w2o4*_ga*MTc1ODQwODQxMS4xNzAxOTAwMTM3*_ga_BRMYJE3DCD*MTcwMjI2MTYxOC4zLjAuMTcwMjI2MTYxOC42MC4wLjA.). 

After downloading the helper Python script and pretrained model as described in **Downloading the Models** section, it is important to ensure the path of model file in the helper script, referenced through `model_path` variable, is correct. 

The model can be compiled from the **Palette docker** using this helper script using command format, `python3 [HELPER_SCRIPT] [PRETRAINED]`. As a sample, for the `resnet101` model downloaded as above from Torchvision, the command outputs may look similar to below.

```
root@59d339853bd1:/home# python3 densenet121.py densenet121.onnx
Model SDK version: 1.1.0
Running calibration ...DONE
2023-12-12 18:38:59,243 - afe.ir.quantization_utils - WARNING - Quantized bias was clipped, resulting in precision loss.  Model may need retraining.
2023-12-12 18:38:59,368 - afe.ir.quantization_utils - WARNING - Quantized bias was clipped, resulting in precision loss.  Model may need retraining.
Running quantization ...DONE

```

After successful compilation, the resulting files are generated in `result/[MODEL_NAME_CALIBRATION_OPTIONS]/mpk` folder which now has `*.yaml, *.json, *.lm` generated as outputs of compilation. These files together can be used for performance estimation as described in next section.

```
root@59d339853bd1:/home# ls
debug.log             densenet121.onnx  densenet121_stage1_mla.lm          docker  root
densenet121_mpk.json  densenet121.py    densenet121_stage1_mla_stats.yaml  result  torchvision_to_onnx.py
root@59d339853bd1:/home# cd result/
root@59d339853bd1:/home/result# ls
densenet121_asym_True_per_ch_True  
root@59d339853bd1:/home/result# cd densenet121_asym_True_per_ch_True/
root@59d339853bd1:/home/result/densenet121_asym_True_per_ch_True# ls
mpk
root@59d339853bd1:/home/result/densenet121_asym_True_per_ch_True# cd mpk
root@59d339853bd1:/home/result/densenet121_asym_True_per_ch_True/mpk# ls
densenet121_mpk.tar.gz
root@59d339853bd1:/home/result/densenet121_asym_True_per_ch_True/mpk# tar zxvf densenet121_mpk.tar.gz 
densenet121_mpk.json
densenet121_stage1_mla_stats.yaml
densenet121_stage1_mla.lm
root@59d339853bd1:/home/result/densenet121_asym_True_per_ch_True/mpk# ls
densenet121_mpk.json  densenet121_mpk.tar.gz  densenet121_stage1_mla.lm  densenet121_stage1_mla_stats.yaml
```



# Performance Estimation #

Using the compiled model using steps above or a precompiled model provided in the repository, it is possible to measure the frames-per-second metric using an actual SiMa.ai developer kit. This requires the developer kit be properly configured as described in **Accelerator Mode** chapter of Palette user guide.

```
python3 accelerator-mode-demos/devkit_inference_models/network_eval/network_eval.py --model_file_path accelerator-modedemos/
devkit_inference_models/model_files/model_files/densenet121/densenet121_MLA_0.lm --mpk_json_path
accelerator-mode-demos/devkit_inference_models/model_files/model_files/densenet121/
densenet121.json --dv_host 192.168.135.170 --image_size 224 224 3
```

Upon running the command the FPS value will show up as below

```

Creating the Forwarding from host
Attempt 0 to forward local port 8000 to 8000 with cmd ssh -f -N -L 8000:localhost:8000
sima@192.168.135.170
Attempt 0 successful to forward local port 8000 to 8000 with pid 613010
Copying the model files to DevKit
Attempt 0 successful to scp
FPS = 272
FPS = 273
FPS = 273
FPS = 274
FPS = 274

```

# License #
The primary license for the models in the [SiMa.ai Model Zoo](https://github.com/SiMa-ai/models) is the BSD 3-Clause License, see [LICENSE](LICENSE.txt). However:



Certain models may be subject to additional restrictions and/or terms. To the extent a LICENSE.txt file is provided for a particular model, please review the LICENSE.txt file carefully as you are responsible for your compliance with such restrictions and/or terms.

