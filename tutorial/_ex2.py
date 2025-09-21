from gst_helper import gst_grouped_frames, atomic_save_image, timing_marks_to_str
import time, argparse

parser = argparse.ArgumentParser(description='GStreamer -> Python RGB frames')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
args, unknown = parser.parse_known_args()

PIPELINE = (
    # Video source
    f"{args.video_source} ! "
    # Properties for the video source
    "video/x-raw,width=1920,height=1080 ! "
    # An identity element so we can track when a new frame is ready (so we can calc. processing time)
    "identity name=frame_ready_webcam silent=false ! "

    # Split the stream
    "tee name=t "

    # Branch A) convert to RGB and send to original appsink
        "t. ! queue max-size-buffers=1 leaky=downstream ! "
        "qtivtransform ! video/x-raw,format=RGB ! "
        "appsink name=original drop=true sync=false max-buffers=1 emit-signals=true "

    # Branch B) resize/crop to 224x224 -> send to another appsink
        "t. ! queue max-size-buffers=1 leaky=downstream ! "
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
    print('    Timings:', timing_marks_to_str(marks))

    # Save image to disk
    frame = frames_by_sink['frame']
    atomic_save_image(frame=frame, path='out/imsdk.png')
    original = frames_by_sink['original']
    atomic_save_image(frame=original, path='out/imsdk_original.png')
