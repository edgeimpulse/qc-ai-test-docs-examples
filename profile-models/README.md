# Profile models

1. Create a `models` directory, and add models in TFLite (`.lite` or `.tflite` extension), ONNX (`.onnx` or `.zip` extension), or SavedModel (`.zip` extension) format.
2. Create a new venv:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt

    wget https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
    pip3 install onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
    ```

3. Create a new model directory and download some models:

    ```bash
    mkdir -p models

    wget -O models/Yolo-X_w8a8.tflite https://huggingface.co/qualcomm/Yolo-X/blob/main/Yolo-X_w8a8.tflite
    wget -O models/Yolo-X_float.tflite https://huggingface.co/qualcomm/Yolo-X/blob/main/Yolo-X_float.tflite
    wget -O models/Yolo-X_w8a8.onnx.zip https://huggingface.co/qualcomm/Yolo-X/blob/main/Yolo-X_w8a8.onnx.zip
    wget -O models/Yolo-X_float.onnx.zip https://huggingface.co/qualcomm/Yolo-X/blob/main/Yolo-X_float.onnx.zip
    ```

3. Profile models:

    ```bash
    python3 profile.py --model-directory models/

    # +----------------------------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | model                            | runtime   | input shape   | variant        | NPU (per inference)   | CPU (per inference)   |
    # +==================================+===========+===============+================+=======================+=======================+
    # | Yolo-X_w8a8.tflite               | tflite    | (1,640,640,3) | quantized_8bit | 26.45ms.              | 298ms.                |
    # +----------------------------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | Yolo-X_w8a8.onnx.zip/model.onnx  | onnx      | (1,3,640,640) | quantized_8bit | FAIL                  | FAIL                  |
    # +----------------------------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | Yolo-X_float.onnx.zip/model.onnx | onnx      | (1,3,640,640) | unquantized    | FAIL                  | 691.1ms.              |
    # +----------------------------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | Yolo-X_float.tflite              | tflite    | (1,640,640,3) | unquantized    | 1163ms.               | 1159ms.               |
    # +----------------------------------+-----------+---------------+----------------+-----------------------+-----------------------+
    ```
