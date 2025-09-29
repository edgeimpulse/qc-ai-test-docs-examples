# Profile models

This has been tested on QCS6490-based development boards (RB3 Gen 2 Vision Kit, Rubik Pi 3) running Ubuntu 24.04.

1. Create a `models` directory, and add models in [TFLite](https://qc-ai-test.gitbook.io/qc-ai-test-docs/running-building-ai-models/lite-rt) (`.lite` or `.tflite` extension), [ONNX](https://qc-ai-test.gitbook.io/qc-ai-test-docs/running-building-ai-models/onnxruntime) (`.onnx` or `.zip` extension), or [context binary](https://qc-ai-test.gitbook.io/qc-ai-test-docs/running-building-ai-models/context-binaries) (`.bin` extension) format.
2. Create a new venv and install packages:

    ```bash
    # Create a new venv, and install packages
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt

    # onnxruntime with qnn bindings
    wget https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
    pip3 install onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
    rm onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl

    # appbuilder for QNN 2.35.0
    pip3 install wheels/qcs6490_qnn2.35.0/qai_appbuilder-2.38.0-cp312-cp312-linux_aarch64.whl
    ```

3. Create a new model directory and download some models:

    ```bash
    mkdir -p models

    # https://aihub.qualcomm.com/models/inception_v3
    wget -O models/inception_v3-inception-v3-float.onnx.zip https://cdn.edgeimpulse.com/qc-ai-docs/models/inception_v3-inception-v3-float.onnx.zip
    wget -O models/inception_v3-inception-v3-float.tflite https://cdn.edgeimpulse.com/qc-ai-docs/models/inception_v3-inception-v3-float.tflite
    wget -O models/inception_v3-inception-v3-qualcomm_qcs6490_proxy-w8a8.onnx.zip https://cdn.edgeimpulse.com/qc-ai-docs/models/inception_v3-inception-v3-qualcomm_qcs6490_proxy-w8a8.onnx.zip
    wget -O models/inception_v3-inception-v3-w8a8.tflite https://cdn.edgeimpulse.com/qc-ai-docs/models/inception_v3-inception-v3-w8a8.tflite

    # https://aiot.aidlux.com/en/models/detail/9?name=inception&precisionShow=1&soc=2
    wget -O models/inception_v3_w8a8.qcs6490.qnn216.ctx.bin https://cdn.edgeimpulse.com/qc-ai-docs/models/inception_v3_w8a8.qcs6490.qnn216.ctx.bin
    ```

3. Profile models:

    ```bash
    python3 profile.py --model-directory models/

    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    # | model                                                                     | runtime        | input shape   | variant        | NPU (per inference)   | CPU (per inference)   |
    # +===========================================================================+================+===============+================+=======================+=======================+
    # | inception_v3-inception-v3-w8a8.tflite                                     | tflite         | (1,224,224,3) | quantized_8bit | 3.377ms.              | 62.04ms.              |
    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    # | inception_v3-inception-v3-qualcomm_qcs6490_proxy-w8a8.onnx.zip/model.onnx | onnx           | FAIL          | quantized_8bit | FAIL                  | FAIL                  |
    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    # | inception_v3-inception-v3-float.onnx.zip/model.onnx                       | onnx           | (1,3,224,224) | unquantized    | FAIL                  | 161.4ms.              |
    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    # | inception_v3-inception-v3-float.tflite                                    | tflite         | (1,224,224,3) | unquantized    | 231.8ms.              | 231.6ms.              |
    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    # | inception_v3_w8a8.qcs6490.qnn216.ctx.bin                                  | ai_runtime_sdk | ?             | ?              | 21.62ms.              | -                     |
    # +---------------------------------------------------------------------------+----------------+---------------+----------------+-----------------------+-----------------------+
    ```
