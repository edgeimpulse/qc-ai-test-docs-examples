import numpy as np, os, time, sys, urllib.request
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image

use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False

def download_file_if_not_exists(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {path} from {url}...")
        urllib.request.urlretrieve(url, path)
    return path

# Path to your model/label/test image (will be download automatically)
MODEL_PATH = download_file_if_not_exists('models/vit-vit-w8a8.tflite', 'https://cdn.edgeimpulse.com/qc-ai-docs/models/vit-vit-w8a8.tflite')
LABELS_PATH = download_file_if_not_exists('models/vit-vit-labels.txt', 'https://cdn.edgeimpulse.com/qc-ai-docs/models/vit-vit-labels.txt')
IMAGE_PATH = download_file_if_not_exists('images/boa-constrictor.jpg', 'https://cdn.edgeimpulse.com/qc-ai-docs/examples/boa-constrictor.jpg')

# Parse labels file
with open(LABELS_PATH, 'r') as f:
    labels = [line for line in f.read().splitlines() if line.strip()]

# Use HTP backend of libQnnTFLiteDelegate.so (NPU) when --use-npu is passed in
experimental_delegates = []
if use_npu:
    experimental_delegates = [load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]

# Load TFLite model and allocate tensors
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load, preprocess and quantize image
def load_image(path, input_shape):
    # Expected input shape: [1, height, width, channels]
    _, height, width, channels = input_shape

    # Load image
    img = Image.open(path).convert("RGB").resize((width, height))
    img_np = np.array(img, dtype=np.float32)
    # !! Normalize... this model is 0..1 scaled (no further normalization); but that depends on your model !!
    img_np = img_np / 255
    # Add batch dim
    img_np = np.expand_dims(img_np, axis=0)

    scale, zero_point = input_details[0]['quantization']  # (scale, zero_point); scale==0.0 -> unquantized

    # Quantize input if needed
    if input_details[0]['dtype'] == np.float32:
        return img_np
    elif input_details[0]['dtype'] == np.uint8:
        # q = round(x/scale + zp)
        q = np.round(img_np / scale + zero_point)
        return np.clip(q, 0, 255).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        # Commonly zero_point ≈ 0 (symmetric), but use provided zp anyway
        q = np.round(img_np / scale + zero_point)
        return np.clip(q, -128, 127).astype(np.int8)
    else:
        raise Exception('Unexpected dtype: ' + str(input_details[0]['dtype']))

input_shape = input_details[0]['shape']
input_data = load_image(IMAGE_PATH, input_shape)

# Set tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run once to warmup
interpreter.invoke()

# Then run 10x
start = time.perf_counter()
for i in range(0, 10):
    interpreter.invoke()
end = time.perf_counter()

# Get prediction
q_output = interpreter.get_tensor(output_details[0]['index'])
scale, zero_point = output_details[0]['quantization']
f_output = (q_output.astype(np.float32) - zero_point) * scale

# Image classification models in AI Hub miss a Softmax() layer at the end of the model, so add it manually
def softmax(x, axis=-1):
    # subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# show top-5 predictions
scores = softmax(f_output[0])
top_k = scores.argsort()[-5:][::-1]
print("\nTop-5 predictions:")
for i in top_k:
    print(f"Class {labels[i]}: score={scores[i]}")

print('')
print(f'Inference took (on average): {((end - start) * 1000) / 10:.4g}ms. per image')
