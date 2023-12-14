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

SiMa.ai subset of compatible models references repositories like Torchvison,
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

