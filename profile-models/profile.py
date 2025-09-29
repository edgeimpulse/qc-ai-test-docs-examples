import time, argparse, os, onnx, zipfile
from helpers import is_onnx_int8_quantized, quantize_onnx_to_int8_nodata_v2, is_tflite_quantized, \
    convert_zipped_savedmodel_to_tflite, has_qnn_delegate, run_perf_tflite, run_perf_onnx, \
    get_input_shape_tflite, get_input_shape_onnx, run_perf_qnncontext
from tabulate import tabulate
from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)

parser = argparse.ArgumentParser(description='Profile models')
parser.add_argument('--model-directory', type=str, required=True, help='Directory with models (.onnx, .tflite, .lite or .zip)')
args, unknown = parser.parse_known_args()

if not os.path.exists(args.model_directory):
    print(f'Model directory ({args.model_directory}, from --model-directory) does not exist')
    exit(1)

qnn_delegate_exists, qnn_delegate_path = has_qnn_delegate()

if not qnn_delegate_exists:
    print('libQnnTFLiteDelegate.so could not be found')
    exit(1)

# Set up the QNN config (/usr/lib => where all QNN libraries are installed)
QNNConfig.Config('/usr/lib', Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)

converted_dir = os.path.join(args.model_directory, 'converted')
os.makedirs(converted_dir, exist_ok=True)

models = []

# TODO: SavedModel conversion hangs... Maybe we just skip all this.

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
                # find the common root (e.g. "model.onnx/")
                members = zf.namelist()
                common_prefix = os.path.commonprefix(members)
                # make sure it's really a directory name
                if not common_prefix.endswith('/'):
                    common_prefix = os.path.dirname(common_prefix) + '/'

                for member in members:
                    # strip the prefix
                    target = member[len(common_prefix):]
                    if not target:  # skip the top-level folder itself
                        continue
                    target_path = os.path.join(unzipped_folder, target)

                    # create any needed directories
                    if member.endswith('/'):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, 'wb') as f:
                            f.write(zf.read(member))
            print('OK')

        unzipped_files = os.listdir(unzipped_folder)

        onnx_file_found = False
        if any(f.endswith(".onnx") for f in unzipped_files):
            for f in unzipped_files:
                if f.endswith('.onnx'):
                    model = os.path.join(model, f)
                    model_path = os.path.join(unzipped_folder, f)
                    onnx_file_found = True

        if not onnx_file_found:
            print('    Cannot find .onnx file in ZIP file, skipping')
            continue

    if model.lower().endswith('.tflite') or model.lower().endswith('.lite'):
        print(f'    Found {model}')

        if is_tflite_quantized(model_path):
            models.append({ 'name': model, 'type': 'tflite', 'quantized_8bit': model_path })
        else:
            models.append({ 'name': model, 'type': 'tflite', 'unquantized': model_path })

    elif model.lower().endswith('.onnx'):
        print(f'    Found {model}, checking quantization... ', end='')

        if is_onnx_int8_quantized(model_path):
            print('OK (quantized)')
            models.append({
                'name': model,
                'type': 'onnx',
                'quantized_8bit': model_path,
            })
        else:
            print('OK (not quantized)')
            models.append({
                'name': model,
                'type': 'onnx',
                'unquantized': model_path,
            })

    elif model.lower().endswith('.bin'):
        print(f'    Found {model}')

        models.append({ 'name': model, 'type': 'ai_runtime_sdk', 'unknown_bin': model_path })

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
                print(f'        NPU: {time_per_inference_ms:.4g}ms')
                row.append(f"{time_per_inference_ms:.4g}ms.")
            else:
                row.append('-')

            time_per_inference_ms = run_perf_tflite(path, use_npu=False)
            print(f'        CPU: {time_per_inference_ms:.4g}ms')
            row.append(f"{time_per_inference_ms:.4g}ms.")

        elif model_type == 'onnx':
            try:
                input_shape, input_type = get_input_shape_onnx(path, load_qnn_htp=True)
            except Exception as e:
                print('get_input_shape_onnx load_qnn_htp=True failed:', e)
                try:
                    input_shape, input_type = get_input_shape_onnx(path, load_qnn_htp=False)

                except Exception as e2:
                    print('get_input_shape_onnx load_qnn_htp=False failed:', e2)
                    row.append('FAIL') # input_shape
                    row.append(variant) # variant
                    row.append('FAIL') # NPU
                    row.append('FAIL') # CPU
                    table.append(row)
                    continue

            row.append(input_shape)
            row.append(variant)

            if qnn_delegate_exists:
                try:
                    time_per_inference_ms = run_perf_onnx(path, use_npu=True)
                    print(f'        NPU: {time_per_inference_ms:.4g}ms')
                    row.append(f"{time_per_inference_ms:.4g}ms.")
                except Exception as e:
                    print(e)
                    row.append('FAIL')
            else:
                row.append('-')

            try:
                time_per_inference_ms = run_perf_onnx(path, use_npu=False)
                print(f'        CPU: {time_per_inference_ms:.4g}ms')
                row.append(f"{time_per_inference_ms:.4g}ms.")
            except Exception as e:
                print(e)
                row.append('FAIL')

        elif model_type == 'ai_runtime_sdk':
            row.append('?') # input shape, https://github.com/quic/ai-engine-direct-helper/issues/24
            row.append('?') # variant https://github.com/quic/ai-engine-direct-helper/issues/24

            time_per_inference_ms = run_perf_qnncontext(path)
            print(f'        NPU: {time_per_inference_ms:.4g}ms')
            row.append(f"{time_per_inference_ms:.4g}ms.")

            row.append('-') # no CPU support

        else:
            raise Exception(f'Invalid type "{model_type}"')

        table.append(row)

print(tabulate(table, ['model', 'runtime', 'input shape', 'variant', 'NPU (per inference)', 'CPU (per inference)'], tablefmt="grid"))
