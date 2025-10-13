from gst_helper import gst_grouped_frames, atomic_save_image, timing_marks_to_str, download_file_if_needed, softmax
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
    imsdk_labels_content = []
    for i in range(0, len(labels)):
        label = labels[i]
        label = label.replace(' ', '-') # no space allowed
        label = label.replace("'", '') # no ' allowed
        imsdk_labels_content.append(f'(structure)"{label},id=(guint){hex(i)},color=(guint)0x00FF00FF;"')
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
    # run inference (using the QNN delegates to run on NPU)
    f'qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options="QNNExternalDelegate,backend_type=htp;" model="{MODEL_PATH}" ! '
    # Run the classifier (add softmax, as AI Hub models miss it), this will return the top n labels (above threshold, min. threshold is 10)
    # note that you also need to pass the quantization params (see below under the "gst_grouped_frames" call).
    f'qtimlvclassification name=cls module=mobilenet extra-operation=softmax threshold=10 results=1 labels="{IMSDK_LABELS_PATH}" ! '

    # Event when inference is done
    "identity name=inference_done silent=false ! "

    # The qtimlvclassification can either output a video/x-raw,format=BGRA,width=224,height=224 element (overlay),
    # or a text/x-raw element (raw text) - here we want the image
    "video/x-raw,format=BGRA,width=224,height=224 ! "

    # Send to application
    "queue max-size-buffers=2 leaky=downstream ! "
    'appsink name=qtimlvclassification_image drop=true sync=false max-buffers=1 emit-signals=true '
)

for frames_by_sink, marks in gst_grouped_frames(PIPELINE, element_properties={
    # the qtimlvclassification element does not like these variables passed in as a string in the pipeline, so set them like this
    'cls': { 'constants': f'Mobilenet,q-offsets=<{zero_point}>,q-scales=<{scale}>' }
}):
    print(f"Frame ready")
    print('    Data:', end='')
    for key in list(frames_by_sink):
        print(f' name={key} {frames_by_sink[key].shape} ({frames_by_sink[key].dtype})', end='')
    print('')

    # Grab the qtimlvclassification_image
    frame = frames_by_sink['qtimlvclassification_image']
    atomic_save_image(frame=frame, path='out/qtimlvclassification_image.png')

    print('    Timings:', timing_marks_to_str(marks))
