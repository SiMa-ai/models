import os
import argparse
import torch
from torchvision.models import *
import torch.onnx

def convert_to_onnx(model_name):
    model = torch.hub.load('pytorch/vision:v0.16.0', model_name, pretrained=True)
    batch_size = 1 # Size of the batch processing
    input_shape = (3, 224, 224) # Input data. Replace it with the actual shape data that model reauires.
    model.eval()
    m_input = torch.randn(batch_size, *input_shape) # Define the input shape.
    print("Before torch.onnx.export", m_input)
    torch.onnx.export(model,
                    m_input,
                    f"{model_name}.onnx",
                    export_params=True,
                    input_names = ["modelInput"], # Construct the input name.
                    output_names = ["modelOutput"], # Construct the output name.
                    opset_version=13, # the ONNX version to export the model to
                    #dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}}) # Dynamic axes of the output is supported.
                            )
    print("After torch.onnx.export")

def main(args):
    convert_to_onnx(args.model_name)

if __name__ == "__main__":
    """ This utility will help you to convert PyTorch/Torchvision models to .onnx.
    Usage : python3 torchvision_to_onnx.py --model_name torchvisioin model name
    On successful model conversion it generates model_name.onnx
    """
    parser = argparse.ArgumentParser(description='PyTorch model name')
    parser.add_argument('--model_name', required=True, type=str, help='PyTorch/Torchvision model name')
    args = parser.parse_args()
    main(args)
