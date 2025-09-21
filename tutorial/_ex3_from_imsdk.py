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

# IM SDK expects labels in this format
# (structure)"white-shark,id=(guint)0x3,color=(guint)0x00FF00FF;" (so no spaces in the name)
IMSDK_LABELS_PATH = 'models/SqueezeNet-1.1_imsdk_labels.txt'
with open(IMSDK_LABELS_PATH, 'w') as f:
    imsdk_labels_content = ['(structure)"background,id=(guint)0x0,color=(guint)0x00FF00FF;"']
    for i in range(0, len(labels)):
        label = labels[i]
        label = label.replace(' ', '-') # no space allowed
        label = label.replace("'", '') # no ' allowed
        imsdk_labels_content.append(f'(structure)"{label},id=(guint){hex(i+1)},color=(guint)0x00FF00FF;"')
    f.write('\n'.join(imsdk_labels_content))

# Load TFLite model and allocate tensors, note: this is a 224x224 model with uint8 input!
# If your models are different, then you'll need to update the pipeline below.
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]     # Use NPU
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = output_details[0]['quantization']

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
    # then resize to 224x224, (!! NOTE: here you need to use format=NV12 to get a tightly packed buffer - if you use RGB this won't work !!)
    "video/x-raw,width=224,height=224,format=NV12 ! "
    # Event when the crop/scale are done
    "identity name=transform_done silent=false ! "
    # turn into right format (UINT8 data type) and add batch dimension
    'qtimlvconverter ! neural-network/tensors,type=UINT8,dimensions=<<1,224,224,3>> ! '
    # Event when conversion is done
    "identity name=conversion_done silent=false ! "
    # run inference (using the QNN delegates to run on NPU)
    f'qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options="QNNExternalDelegate,backend_type=htp;" model="{MODEL_PATH}" ! '
    # Event when inference is done
    "identity name=inference_done silent=false ! "

    # Split the stream
    "tee name=t "

    # Branch A) send raw results to the appsink (note that these are still quantized!)
        "t. ! queue max-size-buffers=1 leaky=downstream ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "appsink name=qtimltflite_output drop=true sync=false max-buffers=1 emit-signals=true "

    # Branch B) parse the output tensor in IM SDK, and then draw the top result onto the earlier image
        "t. ! queue max-size-buffers=1 leaky=downstream ! "
        f'qtimlvclassification module=mobilenet extra-operation=softmax constants="mobilenet,q-offsets=<{zero_point}>,q-scales=<{scale};>" threshold=10 results=2 labels="{IMSDK_LABELS_PATH}" '
        "queue max-size-buffers=2 leaky=downstream ! "
        'appsink name=qtimlvclassification_output drop=true sync=false max-buffers=1 emit-signals=true '
)

for frames_by_sink, marks in gst_grouped_frames(PIPELINE):
    print(f"Frame ready")
    print('    Data:', end='')
    for key in list(frames_by_sink):
        print(f' name={key} {frames_by_sink[key].shape}', end='')
    print('')

    # Get prediction (these come in quantized, so dequantize first)
    q_output = frames_by_sink['qtimltflite_output']
    f_output = (q_output.astype(np.float32) - zero_point) * scale

    # Image classification models in AI Hub miss a Softmax() layer at the end of the model, so add it manually
    scores = softmax(f_output)
    top_k = scores.argsort()[-5:][::-1]
    print(f"    Top-5 predictions:")
    for i in top_k:
        print(f"        Class {labels[i]}: score={scores[i]}")

    print('    Timings:', timing_marks_to_str(marks))
