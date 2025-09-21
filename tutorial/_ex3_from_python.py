from gst_helper import gst_grouped_frames, atomic_save_pillow_image, timing_marks_to_str, download_file_if_needed, softmax
import time, argparse, numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import ImageDraw, Image

parser = argparse.ArgumentParser(description='GStreamer -> SqueezeNet')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
args, unknown = parser.parse_known_args()

MODEL_PATH = download_file_if_needed('models/squeezenet1_1-squeezenet-1.1-w8a8.tflite', 'https://cdn.edgeimpulse.com/qc-ai-docs/models/squeezenet1_1-squeezenet-1.1-w8a8.tflite')
LABELS_PATH = download_file_if_needed('models/SqueezeNet-1.1_labels.txt', 'https://cdn.edgeimpulse.com/qc-ai-docs/models/SqueezeNet-1.1_labels.txt')

# Parse labels
with open(LABELS_PATH, 'r') as f:
    labels = [line for line in f.read().splitlines() if line.strip()]

# Load TFLite model and allocate tensors, note: this is a 224x224 model with uint8 input!
# If your models are different, then you'll need to update the pipeline below.
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]     # Use NPU
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

PIPELINE = (
    # Video source
    f"{args.video_source} ! "
    # Properties for the video source
    "video/x-raw,width=1920,height=1080 ! "
    # An identity element so we can track when a new frame is ready (so we can calc. processing time)
    "identity name=frame_ready_webcam silent=false ! "
    # Crop (square), the crop syntax is ('<X, Y, WIDTH, HEIGHT >').
    # So here we use 1920x1080 input, then center crop to 1080x1080 ((1920-1080)/2 = 420 = x crop)
    f'qtivtransform crop="<420, 0, 1080, 1080>" ! '
    # then resize to 224x224
    "video/x-raw,format=RGB,width=224,height=224 ! "
    # Event when the crop/scale are done
    "identity name=transform_done silent=false ! "
    # Send out the resulting frame to an appsink (where we can pick it up from Python)
    "queue max-size-buffers=2 leaky=downstream ! "
    "appsink name=frame drop=true sync=false max-buffers=1 emit-signals=true "
)

for frames_by_sink, marks in gst_grouped_frames(PIPELINE):
    print(f"Frame ready")
    print('    Data:', end='')
    for key in list(frames_by_sink):
        print(f' name={key} {frames_by_sink[key].shape}', end='')
    print('')

    # Begin inference timer
    inference_start = time.perf_counter()

    # Set tensor with the image received in "frames_by_sink['frame']", add batch dim, and run inference
    interpreter.set_tensor(input_details[0]['index'], frames_by_sink['frame'].reshape((1, 224, 224, 3)))
    interpreter.invoke()

    # Get prediction (dequantized)
    q_output = interpreter.get_tensor(output_details[0]['index'])
    scale, zero_point = output_details[0]['quantization']
    f_output = (q_output.astype(np.float32) - zero_point) * scale

    # Image classification models in AI Hub miss a Softmax() layer at the end of the model, so add it manually
    scores = softmax(f_output[0])

    # End inference timer
    inference_end = time.perf_counter()

    # Add an extra mark, so we have timing info for the complete pipeline
    marks['inference_done'] = list(marks.items())[-1][1] + (inference_end - inference_start)

    # Print top-5 predictions
    top_k = scores.argsort()[-5:][::-1]
    print(f"    Top-5 predictions:")
    for i in top_k:
        print(f"        Class {labels[i]}: score={scores[i]}")

    # Image composition timer
    image_composition_start = time.perf_counter()

    # Add the top 5 scores to the image, and save image to disk (for debug purposes)
    frame = frames_by_sink['frame']
    img = Image.fromarray(frame)
    img_draw = ImageDraw.Draw(img)
    img_draw.text((10, 10), f"{labels[top_k[0]]} ({scores[top_k[0]]:.2f})", fill="black")
    atomic_save_pillow_image(img=img, path='out/ex3_from_python.png')

    image_composition_end = time.perf_counter()

    # Add an extra mark, so we have timing info for the complete pipeline
    marks['image_composition_end'] = list(marks.items())[-1][1] + (image_composition_end - image_composition_start)

    print('    Timings:', timing_marks_to_str(marks))
