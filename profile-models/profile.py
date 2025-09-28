import time, argparse, os, onnx, zipfile
from helpers import is_onnx_int8_quantized, quantize_onnx_to_int8_nodata_v2, is_tflite_quantized, \
    convert_zipped_savedmodel_to_tflite, has_qnn_delegate, run_perf_tflite, run_perf_onnx, \
    get_input_shape_tflite, get_input_shape_onnx
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Profile models')
parser.add_argument('--model-directory', type=str, required=True, help='Directory with models (.onnx, .tflite, .lite or .zip)')
args, unknown = parser.parse_known_args()

if not os.path.exists(args.model_directory):
    print(f'Model directory ({args.model_directory}, from --model-directory) does not exist')
    exit(1)

qnn_delegate_exists, qnn_delegate_path = has_qnn_delegate()

if not qnn_delegate_exists:
    print('[WARN] libQnnTFLiteDelegate.so could not be found, not profiling QNN')
    print('')

converted_dir = os.path.join(args.model_directory, 'converted')
os.makedirs(converted_dir, exist_ok=True)

models = []

# TODO: SavedModel conversion hangs...

print(f'Scanning model directory ({args.model_directory})...')
for model in os.listdir(args.model_directory):
    model_path = os.path.join(args.model_directory, model)
    if os.path.isdir(model_path): continue

    if model.lower().endswith('.zip'):
        print(f'    Found {model}')
        unzipped_folder = os.path.join(converted_dir, model)
        if not os.path.exists(unzipped_folder):
            print(f'         Extracting... ', end='')
            with zipfile.ZipFile(model_path) as zf:
                zf.extractall(unzipped_folder)
            print('OK')

        unzipped_files = os.listdir(unzipped_folder)

        # SavedModel
        if any(f.endswith(".pb") for f in unzipped_files):
            model_f32 = os.path.join(converted_dir, model + '_unquantized.tflite')
            model_i8 = os.path.join(converted_dir, model + '_i8.tflite')

            if not os.path.exists(model_f32) or not os.path.exists(model_i8):
                print('        Converting to tflite... ', end='')
                convert_zipped_savedmodel_to_tflite(unzipped_folder, model_f32, model_i8)
                print('OK')

            models.append({
                'name': model,
                'type': 'tflite',
                'unquantized': model_f32,
                'quantized_8bit': model_i8,
            })
            continue

        elif any(f.endswith(".onnx") for f in unzipped_files):
            for f in unzipped_files:
                if f.endswith('.onnx'):
                    model = os.path.join(model, f)
                    model_path = os.path.join(unzipped_folder, f)

    if model.lower().endswith('.tflite') or model.lower().endswith('.lite'):
        print(f'    Found {model}')

        if is_tflite_quantized(model_path):
            models.append({ 'name': model, 'type': 'tflite', 'quantized_8bit': model_path })
        else:
            models.append({ 'name': model, 'type': 'tflite', 'unquantized': model_path })

    elif model.lower().endswith('.onnx'):
        print(f'    Found {model}, checking quantization... ', end='')

        if not is_onnx_int8_quantized(model_path):
            model_path_i8 = os.path.join(converted_dir, model.lower().replace('.onnx', '_i8.onnx'))
            if not os.path.exists(model_path_i8):
                print('OK, not quantized yet')
                print('        Quantizing... ', end='')
                quantize_onnx_to_int8_nodata_v2(model_path, model_path_i8)
                print('OK')
            else:
                print(f'OK, quantized version already exists in {model_path_i8}')

            models.append({
                'name': model,
                'type': 'onnx',
                'unquantized': model_path,
                'quantized_8bit': model_path_i8,
            })
        else:
            print('OK, already quantized')
            models.append({
                'name': model,
                'type': 'onnx',
                'quantized_8bit': model_path,
            })
print('')

print('Testing models...')
table = []
for model in models:
    model_name = model['name']
    model_type = model['type']
    for variant in model.keys():
        if variant == 'name': continue
        if variant == 'type': continue
        path = model[variant]
        print(f'    Running {model_name} ({variant}, {model_type}): {path}...')

        row = [ model_name, model_type ]

        if model_type == 'tflite':
            input_shape, input_type = get_input_shape_tflite(path)
            row.append(input_shape)
            row.append(variant)

            if qnn_delegate_exists:
                time_per_inference_ms = run_perf_tflite(path, use_npu=True)
                print(f'        NPU: {time_per_inference_ms}ms')
                row.append(f"{time_per_inference_ms:.4g}ms.")
            else:
                row.append('-')

            time_per_inference_ms = run_perf_tflite(path, use_npu=False)
            print(f'        CPU: {time_per_inference_ms}ms')
            row.append(f"{time_per_inference_ms:.4g}ms.")

        elif model_type == 'onnx':
            input_shape, input_type = get_input_shape_onnx(path)
            row.append(input_shape)
            row.append(variant)

            if qnn_delegate_exists:
                time_per_inference_ms = run_perf_onnx(path, use_npu=True)
                print(f'        NPU: {time_per_inference_ms}ms')
                row.append(f"{time_per_inference_ms:.4g}ms.")
            else:
                row.append('-')

            time_per_inference_ms = run_perf_onnx(path, use_npu=False)
            print(f'        CPU: {time_per_inference_ms}ms')
            row.append(f"{time_per_inference_ms:.4g}ms.")

        else:
            raise Exception(f'Invalid type "{model_type}"')

        table.append(row)

print(tabulate(table, ['model', 'runtime', 'input shape', 'variant', 'NPU (per inference)', 'CPU (per inference)'], tablefmt="grid"))
