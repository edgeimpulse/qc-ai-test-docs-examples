# Profile models

1. Create a `models` directory, and add models in TFLite (`.lite` or `.tflite` extension), ONNX (`.onnx` or `.zip` extension), or SavedModel (`.zip` extension) format.
2. Create a new venv:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```

3. Profile models:

    ```bash
    python3 profile.py --model-directory models/

    # +-----------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | model           | runtime   | input shape   | variant        | NPU (per inference)   | CPU (per inference)   |
    # +=================+===========+===============+================+=======================+=======================+
    # | yolov5.onnx     | onnx      | (1,3,160,160) | unquantized    | -                     | 1.493ms.              |
    # +-----------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | yolov5.onnx     | onnx      | (1,3,160,160) | quantized_8bit | -                     | 1.012ms.              |
    # +-----------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | yolov5_i8.lite  | tflite    | (1,160,160,3) | quantized_8bit | -                     | 1.178ms.              |
    # +-----------------+-----------+---------------+----------------+-----------------------+-----------------------+
    # | yolov5_f32.lite | tflite    | (1,160,160,3) | unquantized    | -                     | 3.27ms.               |
    # +-----------------+-----------+---------------+----------------+-----------------------+-----------------------+
    ```
