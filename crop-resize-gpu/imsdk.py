from gst_helper import gst_grouped_frames, atomic_save_image, timing_marks_to_str
import time, argparse

parser = argparse.ArgumentParser(description='GStreamer -> Python RGB frames')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
args, unknown = parser.parse_known_args()

video_source_width = 1920
video_source_height = 1080

# we want to center crop, so calculate the no. of pixels
crop_x = int((video_source_width - video_source_height) / 2)
crop_y = 0
crop_width = video_source_height
crop_height = video_source_height

PIPELINE = (
    # Video source
    f"{args.video_source} ! "
    # Properties for the video source
    f"video/x-raw,width={video_source_width},height={video_source_height} ! "
    # An identity element so we can track when a new frame is ready (so we can calc. processing time)
    "identity name=frame_ready silent=false ! "
    #  then scale to 224x224
    # Crop (square), the crop syntax is ('<X, Y, WIDTH, HEIGHT >').
    f'qtivtransform crop="<{crop_x}, {crop_y}, {crop_width}, {crop_height}>" ! '
    # then resize to 224x224
    "video/x-raw,format=RGB,width=224,height=224 ! "
    # Event when the crop/scale are done
    "identity name=transform_done silent=false ! "
    # Send out the resulting frame to an appsink (where we can pick it up from Python)
    "queue max-size-buffers=2 leaky=downstream ! "
    "appsink name=frame drop=true sync=false max-buffers=1 emit-signals=true"
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
