# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# **************************************************************************
# Author: Sumit Mishra (sumit.mishra@sima.ai)

import os
import numpy as np
import dataclasses

from afe.load.importers.general_importer import ImporterParams, pytorch_source
from afe.apis.defines import QuantizationParams, quantization_scheme, default_calibration, CalibrationMethod
from afe.apis.release_v1 import get_model_sdk_version
from afe.apis.loaded_net import load_model
from afe.apis.model import Model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.backends.mpk.interface import L2CachingMode
from afe.ir.tensor_type import scalar_type_from_dtype, scalar_type_to_dtype


# Helper function to construct ImporterParams for the model to be loaded
def _get_import_params(*args, **kwargs):
    framework = "pytorch"
    layout = "NCHW"
    if framework == "onnx" and layout == "NHWC":
        from afe.load.importers.general_importer import ModelFormat
        model_path = kwargs["model_path"]
        shape_dict = kwargs["shape_dict"]
        dtype_dict = kwargs["dtype_dict"]
        input_names = list()
        input_shape = list()
        input_type = list()
        for k, v in shape_dict.items():
            input_names.append(k)
            input_shape.append(v)

        for v in dtype_dict.values():
            input_type.append(v)

        params = ImporterParams(format=ModelFormat.onnx,
                                file_paths=[model_path],
                                input_names=input_names,
                                input_shapes=input_shape,
                                input_types=input_type,
                                layout=layout)
    else:
        params = pytorch_source(*args, **kwargs)
    return params


def main(arm_only, asym, per_ch, calibration,
         retain, l2_cache, verbose, load_net):
    if verbose:
        enable_verbose_error_messages()
    # Get Model SDK version
    sdk_version = get_model_sdk_version()
    print(f"Model SDK version: {sdk_version}")

    # Model information
    input_names = ['input0']
    input_shapes = [[1, 3, 224, 224]]
    input_dtypes = [scalar_type_from_dtype("float32")]
    assert len(input_names) == len(input_shapes)
    args1 = {'model_path': 'models/UR_pt_vgg19_fp32_224_224.pt', 'input_names': ['input0'], 'input_shapes': [[1, 3, 224, 224]]}
    if 'shape_dict' in args.keys():
        shape_dict = {name: shape for name, shape in zip(input_names, input_shapes)}
        args['shape_dict'] = shape_dict
    if 'dtype_dict' in args.keys():
        dtype_dict = {name: dtype for name, dtype in zip(input_names, input_dtypes)}
        args['dtype_dict'] = dtype_dict
    print(args)
    # Load a model and the result is a LoadedNet
    params = _get_import_params(**args)
    loaded_net = load_model(params)

    # reset random seed for input data generation
    np.random.seed(123)

    # Prepare input data
    inputs = dict()
    for i, input_name in enumerate(input_names):
        input_shape = input_shapes[i]
        input_dtype = input_dtypes[i]
        sample_input = np.random.rand(*input_shape).astype(scalar_type_to_dtype(input_dtype))
        inputs[input_name] = sample_input

    layout = "NCHW"
    if layout == "NCHW":
        for k, v in inputs.items():
            inputs[k] = np.transpose(v, [0, 2, 3, 1])

    # Execute the loaded net
    loaded_net_output = loaded_net.execute(inputs)

    #Quntize the loaded net and the result is a quantized SDK Model net
    calibration_data = [inputs]
    quant_configs: QuantizationParams = QuantizationParams(calibration_method=default_calibration(),
                                                           activation_quantization_scheme=quantization_scheme(asym, False),
                                                           weight_quantization_scheme=quantization_scheme(False, per_ch),
                                                           node_names={''},
                                                           custom_quantization_configs=None)
    if calibration in ['mse', 'moving_average', 'entropy', 'percentile']:
        calibration_method = CalibrationMethod.from_str(calibration)
    else:
        calibration_method = None
    if calibration_method:
        quant_configs = dataclasses.replace(quant_configs, calibration_method=calibration_method)
    sdk_net = loaded_net.quantize(
        calibration_data=calibration_data,
        quantization_config=quant_configs,
        model_name="UR_pt_vgg19_fp32_224_224",
        arm_only=arm_only
    )
    # Execute the quantized net
    sdk_net_output = sdk_net.execute(inputs=inputs)

    # Compare outputs of the loaded net and quantized SDK net
    for load_o, sdk_o in zip(loaded_net_output, sdk_net_output):
        max_err = np.max(abs(load_o.astype(np.float32) - sdk_o.astype(np.float32)))
        print(f"Max absolute error between outputs of loaded net and quantized net = {max_err}")
    saved_model_name = f"UR_pt_vgg19_fp32_224_224_asym_{asym}_per_ch_{per_ch}"
    if load_net:
        # Save the SDK net and two files are generated: sima model file and JSON file for Netron
        # Extension ".sima" is added internally if not present in the provided name
        saved_model_directory = os.path.join(os.getcwd(), 'result', saved_model_name, 'sdk_net')
        os.makedirs(saved_model_directory, mode=0o777, exist_ok=True)
        sdk_net.save(model_name=saved_model_name, output_directory=saved_model_directory)

        # Load a saved net - note that sima extention is optional
        load_model_name = f"UR_pt_vgg19_fp32_224_224_asym_{asym}_per_ch_{per_ch}.sima"
        net_read_back = Model.load(model_name=load_model_name, network_directory=saved_model_directory)
        assert isinstance(net_read_back, Model)

    # Compile the quantized net and generate LM file and MPK JSON file
    saved_mpk_directory = os.path.join(os.getcwd(), 'result', saved_model_name, 'mpk')
    os.makedirs(saved_mpk_directory, mode=0o777, exist_ok=True)
    if retain and l2_cache:
        sdk_net.compile(output_path=saved_mpk_directory,
                        batch_size=1,
                        l2_caching_mode=L2CachingMode.SINGLE_MODEL,
                        retained_temporary_directory_name=saved_mpk_directory)
    elif retain:
        sdk_net.compile(output_path=saved_mpk_directory,
                        retained_temporary_directory_name=saved_mpk_directory)
    elif l2_cache:
        sdk_net.compile(output_path=saved_mpk_directory,
                        batch_size=1,
                        l2_caching_mode=L2CachingMode.SINGLE_MODEL)
    else:
        sdk_net.compile(output_path=saved_mpk_directory)


if __name__ == "__main__":
    main(False, True, True, 'min_max', True, False, True, False)
