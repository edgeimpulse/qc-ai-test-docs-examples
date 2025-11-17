from gst_helper import gst_grouped_frames, timing_marks_to_str, mark_performance, OutputStreamer, has_library
from gst_pipeline_helper import get_gstreamer_input_pipeline, get_gstreamer_output_pipeline_mp4
import time, argparse, numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import ImageDraw, Image
from preprocessing import rgb_numpy_arr_to_input_tensor
from postprocessing import face_det_lite_postprocessing

parser = argparse.ArgumentParser(description='Face Detection demo on GPU/NPU')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
parser.add_argument('--video-input-width', type=int, required=False, default=1920, help='Video width (input), default 1920')
parser.add_argument('--video-input-height', type=int, required=False, default=1080, help='Video height (input), default 1080')
parser.add_argument('--resize-mode', type=str, required=False, default='fit-short', help='Crop method (either "squash" or "fit-short")')
args, unknown = parser.parse_known_args()

# Load TFLite model and allocate tensors
interpreter = Interpreter(
    model_path='face_det_lite-lightweight-face-detection-w8a8.tflite',
    # Use NPU if QNN is available
    experimental_delegates=[load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})] if has_library('libQnnTFLiteDelegate.so') else None,
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# TODO: Figure out webcam resolution automatically
# TODO: Also get original resolution out (so we can overlay on original resolution); see ex. here https://qc-ai-test.gitbook.io/qc-ai-test-docs/running-building-ai-models/im-sdk#ex-2-teeing-streams-and-multiple-outputs
# TODO: Allow adding crop mode "fit-long"

# Input pipeline
input_pipeline = get_gstreamer_input_pipeline(args.video_source, args.video_input_width, args.video_input_height,
    resize_mode=args.resize_mode, interpreter=interpreter)

# Output pipeline (write to .mp4 file)
# TODO: Make this configurable through CLI
output_streamer = OutputStreamer(pipeline_tail=get_gstreamer_output_pipeline_mp4('out/out.mp4'))

try:
    for frames_by_sink, marks in gst_grouped_frames(input_pipeline):
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

        output_streamer.push(img_out)

        print('    Timings:', timing_marks_to_str(marks))
finally:
    output_streamer.stop()
