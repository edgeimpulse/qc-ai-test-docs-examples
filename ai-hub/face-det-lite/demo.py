from gst_helper import gst_grouped_frames, atomic_save_pillow_image, timing_marks_to_str, download_file_if_needed, mark_performance
import time, argparse, numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import ImageDraw, Image
from preprocessing import rgb_numpy_arr_to_input_tensor
from postprocessing import face_det_lite_postprocessing

parser = argparse.ArgumentParser(description='Face Detection demo on GPU/NPU')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
args, unknown = parser.parse_known_args()

# Load TFLite model and allocate tensors, note: this is a 224x224 model with uint8 input!
# If your models are different, then you'll need to update the pipeline below.
interpreter = Interpreter(
    model_path='face_det_lite-lightweight-face-detection-w8a8.tflite',
    experimental_delegates=[load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]     # Use NPU
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# TODO: Figure out webcam resolution automatically
# TODO: Figure out crop automatically
# TODO: Also get original resolution out (so we can overlay on original resolution); see ex. here https://qc-ai-test.gitbook.io/qc-ai-test-docs/running-building-ai-models/im-sdk#ex-2-teeing-streams-and-multiple-outputs
# TODO: Allow specifying crop mode (squash / fit-short / fit-long) - now hardcoded to fit-short

# TODO: Abstract this away
PIPELINE = (
    # Video source
    f"{args.video_source} ! "
    # Properties for the video source
    "video/x-raw,width=1920,height=1080 ! "
    # An identity element so we can track when a new frame is ready (so we can calc. processing time)
    "identity name=frame_ready_webcam silent=false ! "
    # Crop (square), the crop syntax is ('<X, Y, WIDTH, HEIGHT >').
    # So here we use 1920x1080 input, then crop to 1440x1080 so we have the same aspect ratio
    f'qtivtransform crop="<240, 0, 1440, 1080>" ! '
    # then resize to 224x224
    f"video/x-raw,format=RGB,width={input_w},height={input_h} ! "
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

    with mark_performance('inference_done', marks):
        # Set tensor with the image received in "frames_by_sink['frame']", add batch dim, and run inference
        frame = frames_by_sink['frame']
        # Facedet lite uses blue channel only
        interpreter.set_tensor(input_details[0]['index'], rgb_numpy_arr_to_input_tensor(interpreter, arr=frame, single_channel_behavior='blue'))
        interpreter.invoke()

    with mark_performance('postprocessing_done', marks):
        # Get prediction (dequantized)
        faces = face_det_lite_postprocessing(interpreter)
        print('    Faces:', faces)

    # Composite image using Pillow
    # TODO: Do this on the original, unresized/cropped image
    if frame.shape[2] == 1:
        frame = np.squeeze(frame, axis=-1) # strip off the last dim if grayscale
    img_out = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img_out)
    for bb in faces:
        L, T, W, H, score = bb
        draw.rectangle([L, T, L + W, T + H], outline="#00FF00", width=3)

    # And write to output image so we can debug
    # TODO: Allow using GStreamer appsrc to pipe this somewhere
    atomic_save_pillow_image(img=img_out, path='out/facedet.png')

    print('    Timings:', timing_marks_to_str(marks))
