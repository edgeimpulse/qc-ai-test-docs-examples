from gst_helper import gst_grouped_frames, atomic_save_image, timing_marks_to_str, download_file_if_needed, softmax
import time, argparse, numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import ImageDraw, Image

parser = argparse.ArgumentParser(description='GStreamer -> SqueezeNet')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
args, unknown = parser.parse_known_args()

if args.video_source.strip() == '':
    raise Exception('--video-source is empty, did you not set the IMSDK_VIDEO_SOURCE env variable? E.g.:\n' +
    '    export IMSDK_VIDEO_SOURCE="v4l2src device=/dev/video2"')

# Source: https://commons.wikimedia.org/wiki/File:Arrow_png_image.png
OVERLAY_IMAGE = download_file_if_needed('images/imsdk-transparent-static.png', 'https://cdn.edgeimpulse.com/qc-ai-docs/example-images/imsdk-transparent-static.png')
OVERLAY_WIDTH = 128
OVERLAY_HEIGHT = 96

PIPELINE = (
    # Part 1: Create a qtivcomposer with two sinks (we'll write webcam to sink 0, overlay to sink 1)
    "qtivcomposer name=comp sink_0::zorder=0 "
        # Sink 1 (the overlay) will be at x=10, y=10; and sized 128x96
        f"sink_1::zorder=1 sink_1::alpha=1.0 sink_1::position=<10,10> sink_1::dimensions=<{OVERLAY_WIDTH},{OVERLAY_HEIGHT}> ! "
    "videoconvert ! "
    "video/x-raw,format=RGBA,width=224,height=224 ! "
    "identity name=composer_done silent=false ! "
    # Write frames to appsink
    "appsink name=overlay_raw drop=true sync=false max-buffers=1 emit-signals=true "

    # Part 2: Grab image from webcam and write the composer
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
        "video/x-raw,width=224,height=224,format=NV12 ! "
        # Event when the crop/scale are done
        "identity name=transform_done silent=false ! "
        # Write to sink 0 on the composer
        "comp.sink_0 "

    # Part 3: Load overlay from disk and write to composer (sink 1)
        # Image (statically from disk)
        f'filesrc location="{OVERLAY_IMAGE}" ! '
        # Decode PNG
        "pngdec ! "
        # Turn into a video (scaled to 128x96, RGBA format so we keep transparency, requires a framerate)
        "imagefreeze ! "
        "videoscale ! "
        "videoconvert ! "
        f"video/x-raw,format=RGBA,width={OVERLAY_WIDTH},height={OVERLAY_HEIGHT},framerate=30/1 ! "
        # Write to sink 1 on the composer
        "comp.sink_1 "
)

for frames_by_sink, marks in gst_grouped_frames(PIPELINE):
    print(f"Frame ready")
    print('    Data:', end='')
    for key in list(frames_by_sink):
        print(f' name={key} {frames_by_sink[key].shape} ({frames_by_sink[key].dtype})', end='')
    print('')

    # Save image to disk
    save_image_start = time.perf_counter()
    frame = frames_by_sink['overlay_raw']
    atomic_save_image(frame=frame, path='out/webcam_with_overlay.png')
    save_image_end = time.perf_counter()

    # Add an extra mark, so we have timing info for the complete pipeline
    marks['save_image_end'] = list(marks.items())[-1][1] + (save_image_end - save_image_start)

    print('    Timings:', timing_marks_to_str(marks))
