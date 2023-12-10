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

from afe.load.importers.general_importer import ModelFormat, ImporterParams, tflite_source
from afe.apis.defines import QuantizationParams, quantization_scheme, default_calibration, CalibrationMethod
from afe.apis.release_v1 import get_model_sdk_version
from afe.apis.loaded_net import load_model
from afe.apis.model import Model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.backends.mpk.interface import L2CachingMode
from afe.ir.defines import InputName
from afe.ir.tensor_type import ScalarType


# Helper function to construct ImporterParams for the model to be loaded
def _get_import_params(*args, **kwargs):
    params = tflite_source(*args, **kwargs)
    return params


def main(arm_only, asym, per_ch, calibration,
         retain, l2_cache, verbose, load_net):
    if verbose:
        enable_verbose_error_messages()
    # Get Model SDK version
    sdk_version = get_model_sdk_version()
    print(f"Model SDK version: {sdk_version}")

    # Model information
    input_names, input_shapes, input_type = ("serving_default_image_input:0", (1, 416, 416, 3), ScalarType.float32)
    args = {'model_path': '', 'shape_dict': '', 'dtype_dict': ''}
    
    if 'shape_dict' in args.keys():
        if type(input_names) is list:
            shape_dict = {input_name: input_shape for input_name, input_shape in zip(input_names,input_shapes)}
        else:
            shape_dict = {input_names: input_shapes}
        args['shape_dict'] = shape_dict
    if 'dtype_dict' in args.keys():
        if type(input_names) is list:
            dtype_dict = {input_name: input_type for input_name in input_names}
        else:
            dtype_dict = {input_names: input_type}
        args['dtype_dict'] = dtype_dict
    if 'model_path' in args.keys():
        if os.path.isfile("openvino/public/yolo-v4-tf/yolo-v4-tf.tflite"):
            args['model_path'] = "openvino/public/yolo-v4-tf/yolo-v4-tf.tflite"
        else:
            args['model_path'] = "/project/jenkinsagents/qa/models/openvino/public/yolo-v4-tf/yolo-v4-tf.tflite"
        
    print(args)
    # Load a model and the result is a LoadedNet
    params = _get_import_params(**args)
    loaded_net = load_model(params)
    if type(input_names) is not list:
        inputs = {InputName(input_names): np.random.rand(1, 416, 416, 3)}
    else:
        inputs = {InputName(input_name): np.random.rand(*input_shape) for input_name,input_shape in zip(input_names,input_shapes)}
    

    # Quantize the loaded net and the result is a quantized SDK Model net
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
        model_name="yolo-v4-tf",
        arm_only=arm_only
    )
    # Execute the quantized net
    sdk_net_output = sdk_net.execute(inputs=inputs)

    saved_model_name = f"yolo-v4-tf_asym_{asym}_per_ch_{per_ch}"
    if load_net:
        # Save the SDK net and two files are generated: sima model file and JSON file for Netron
        # Extension ".sima" is added internally if not present in the provided name
        saved_model_directory = os.path.join(os.getcwd(), 'result', saved_model_name, 'sdk_net')
        os.makedirs(saved_model_directory, mode=0o777, exist_ok=True)
        sdk_net.save(model_name=saved_model_name, output_directory=saved_model_directory)

        # Load a saved net - note that sima extention is optional
        load_model_name = f"yolo-v4-tf_asym_{asym}_per_ch_{per_ch}.sima"
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
    main(False, True, True, 'min_max', False, False, False, False)
