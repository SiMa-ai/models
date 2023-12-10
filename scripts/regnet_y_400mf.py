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

from afe.apis.defines import QuantizationParams, quantization_scheme, default_calibration, CalibrationMethod
from afe.apis.loaded_net import load_model
from afe.apis.model import Model
from afe.apis.release_v1 import get_model_sdk_version
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.backends.mpk.interface import L2CachingMode
from afe.ir.defines import InputName
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source


def main(arm_only, asym, per_ch, calibration, 
         retain, l2_cache, verbose, load_net):
    if verbose:
        enable_verbose_error_messages()
    # Get Model SDK version
    sdk_version = get_model_sdk_version()
    print(f"Model SDK version: {sdk_version}")

    # Model information
    input_name, input_shape, input_type = ("modelInput", (1, 3, 224, 224), ScalarType.float32)
    input_shapes_dict = {input_name: input_shape}
    input_types_dict = {input_name: input_type}

    model_path1 = "torchvision_onnx_models/regnet_y_400mf.onnx"
    model_path2 = "/project/jenkinsagents/qa/models/torchvision_onnx_models/regnet_y_400mf.onnx"
    if os.path.isfile(model_path1):
        model_path = model_path1
    else:
        model_path = model_path2
    importer_params = onnx_source(model_path, input_shapes_dict, input_types_dict)

    loaded_net = load_model(importer_params)

    inputs = {InputName(input_name): np.random.rand(1, 224, 224, 3)}
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

    sdk_net = loaded_net.quantize(calibration_data,
                                  quant_configs,
                                  model_name="regnet_y_400mf",
                                  arm_only=arm_only)
    # Execute the quantized net
    sdk_net_output = sdk_net.execute(inputs=inputs)
        
    saved_model_name = f"regnet_y_400mf_asym_{asym}_per_ch_{per_ch}"
    if load_net:
        # Save the SDK net and two files are generated: sima model file and JSON file for Netron
        # Extension ".sima" is added internally if not present in the provided name
        saved_model_directory = os.path.join(os.getcwd(), 'result', saved_model_name, 'sdk_net')
        os.makedirs(saved_model_directory, mode=0o777, exist_ok=True)
        sdk_net.save(model_name=saved_model_name, output_directory=saved_model_directory)
    
        # Load a saved net - note that sima extention is optional
        load_model_name = f"regnet_y_400mf_asym_{asym}_per_ch_{per_ch}.sima"
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
