import os
import onnx
from onnxruntime.quantization import QuantType
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import os, zipfile
import tensorflow as tf
import pathlib
from ai_edge_litert.interpreter import Interpreter, load_delegate
import time
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, QuantFormat, QuantType, CalibrationMethod
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnxruntime as ort, numpy as np

def has_qnn_delegate():
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in ld_path.split(":"):
        if not path:
            continue
        candidate = pathlib.Path(path) / "libQnnTFLiteDelegate.so"
        if candidate.exists():
            return True, str(candidate)
    return False, None

Q_OPS = {
    "QuantizeLinear", "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinearConv", "QLinearMatMul", "QLinearAdd", "QLinearSigmoid"
}

def is_onnx_int8_quantized(path: str) -> bool:
    model = onnx.load(path)
    # Heuristic 1: Q/DQ or QLinear ops present
    if any(n.op_type in Q_OPS for n in model.graph.node):
        return True
    # Heuristic 2: any int8/uint8 initializers used by known quantized ops
    int_tensors = {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
    if any(init.data_type in int_tensors for init in model.graph.initializer):
        # Not definitive on its own, but usually indicates quantization
        return True
    return False

def quantize_onnx_to_int8_nodata_v2(fp32, int8):
    class RandomData(CalibrationDataReader):
        def __init__(self, model_path, n=20):
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.inputs = self.sess.get_inputs()
            self.n = n
            self.i = 0
        def get_next(self):
            if self.i >= self.n: return None
            self.i += 1
            feeds = {}
            for inp in self.inputs:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                if "float" in inp.type:
                    feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
                else:
                    # most calibrators expect float inputs; adjust if your model expects ints
                    feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
            return feeds

    dr = RandomData(fp32, n=32)

    quantize_static(
        model_input=fp32,
        model_output=int8,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,                  # <-- avoids ConvInteger
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

def is_tflite_quantized(path: str) -> bool:
    """
    Returns True if ANY input tensor looks quantized (uint8/int8/int16 with nonzero scale).
    Uses ai-edge-litert only. No extras, no FlatBuffers parsing.
    """
    interp = Interpreter(model_path=path)
    details = interp.get_input_details()  # list of dicts

    for d in details:
        dt = d["dtype"]
        if dt not in (np.uint8, np.int8, np.int16):
            continue

        # Prefer per-tensor/tensor-wise scales if present
        q = d.get("quantization", (0.0, 0))
        scale = float(q[0]) if isinstance(q, (tuple, list)) and len(q) else 0.0

        qp = d.get("quantization_parameters", {})
        scales = qp.get("scales", [])
        has_scales = hasattr(scales, "__len__") and len(scales) > 0 and any(abs(float(s)) > 0 for s in scales)

        if scale != 0.0 or has_scales:
            return True

    return False

def _convert_tflite_float32(saved_model_dir: str) -> bytes:
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Default = float32; add basic graph optimizations
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()

def _convert_tflite_int8_dynamic(saved_model_dir: str, allow_select_tf_ops: bool=False) -> bytes:
    """
    Dynamic range int8: quantizes weights to int8 without a representative dataset.
    Inputs/outputs usually remain float32 (that’s expected).
    """
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # Hint the converter to prefer int8 kernels when possible
    conv.target_spec.supported_types = [tf.int8]
    if allow_select_tf_ops:
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    else:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()

def convert_zipped_savedmodel_to_tflite(
    sm_dir: str,
    out_fp32_path: str = "model_f32.tflite",
    out_i8_path: str = "model_i8_dynamic.tflite",
    allow_select_tf_ops: bool = False,
) -> tuple[str, str]:
    f32 = _convert_tflite_float32(sm_dir)
    with open(out_fp32_path, "wb") as f:
        f.write(f32)
    i8 = _convert_tflite_int8_dynamic(sm_dir, allow_select_tf_ops=allow_select_tf_ops)
    with open(out_i8_path, "wb") as f:
        f.write(i8)
    return out_fp32_path, out_i8_path

def is_zipped_savedmodel(path: str) -> bool:
    """
    Check if a path (directory or zip file) looks like a TF SavedModel.
    """
    if os.path.isdir(path):
        return (
            os.path.isfile(os.path.join(path, "saved_model.pb")) or
            os.path.isfile(os.path.join(path, "saved_model.pbtxt"))
        ) and os.path.isdir(os.path.join(path, "variables"))

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        has_pb = any("saved_model.pb" in n or "saved_model.pbtxt" in n for n in names)
        has_vars = any(n.startswith("variables/") for n in names)
        return has_pb and has_vars

    return False

def get_input_shape_tflite(model):
    interpreter = Interpreter(
        model_path=model,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    shapes = []
    types = []
    for inp in input_details:
        shapes.append('(' + (','.join([ str(x) for x in inp['shape'] ])) + ')')
        types.append(str(inp['dtype']))

    return ', '.join(shapes), ', '.join(types)

def get_input_shape_onnx(model):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    shapes = []
    types = []
    for inp in sess.get_inputs():
        shapes.append('(' + (','.join([ str(x) for x in inp.shape ])) + ')')
        types.append(str(inp.type))

    return ', '.join(shapes), ', '.join(types)

def run_perf_tflite(model, use_npu):
    delegates = None
    if use_npu:
        delegates = [load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]

    interpreter = Interpreter(
        model_path=model,
        experimental_delegates=delegates,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    # run 5x to warm up
    for i in range(0, 5):
        interpreter.invoke()

    TIMES = 10

    # Then run 10x
    start = time.perf_counter()
    for i in range(0, TIMES):
        interpreter.invoke()
    end = time.perf_counter()

    time_per_inference_ms = ((end - start) * 1000) / TIMES

    return time_per_inference_ms

def run_perf_onnx(model, use_npu):
    providers = []
    if use_npu:
        providers.append(("QNNExecutionProvider", {
            "backend_type": "htp",
        }))
    else:
        providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()

    sess = ort.InferenceSession(model, sess_options=so, providers=providers)

    # create random inputs
    feeds = {}
    for inp in sess.get_inputs():
        # Replace dynamic dims (None) with 1
        shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]

        if "float" in inp.type:
            feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
        elif "int64" in inp.type:
            feeds[inp.name] = np.random.randint(0, 10, size=shape, dtype=np.int64)
        elif "int32" in inp.type:
            feeds[inp.name] = np.random.randint(0, 10, size=shape, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported dtype: {inp.type}")

    # run 5x to warm up
    for i in range(0, 5):
        _ = sess.run(None, feeds)

    TIMES = 10

    # Then run 10x
    start = time.perf_counter()
    for i in range(0, TIMES):
        _ = sess.run(None, feeds)
    end = time.perf_counter()

    time_per_inference_ms = ((end - start) * 1000) / TIMES

    return time_per_inference_ms
