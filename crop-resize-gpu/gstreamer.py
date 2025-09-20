import time, threading, collections, numpy as np
import gi, sys, os, signal, argparse
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
from PIL import Image

Gst.init(None)

parser = argparse.ArgumentParser(description='GStreamer -> Python RGB frames')
parser.add_argument('--video-source', type=str, required=True, help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")')
parser.add_argument('--use-imsdk', action='store_true')
args, unknown = parser.parse_known_args()

if args.use_imsdk:
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
        # Send out the resulting frame to an appsink (where we can pick it up from Python)
        "queue max-size-buffers=2 leaky=downstream ! "
        "appsink name=sink drop=true sync=false max-buffers=1 emit-signals=true"
    )
    PREFIX = '[IMSDK]'
    FILE_NAME = 'imsdk.png'
else:
    # Native GStreamer pipeline
    PIPELINE = (
        # Video source
        f"{args.video_source} ! "
        # Properties for the video source
        "video/x-raw,width=1920,height=1080 ! "
        # An identity element so we can track when a new frame is ready (so we can calc. processing time)
        "identity name=frame_ready silent=false ! "
        # Crop to square
        "videoconvert ! aspectratiocrop aspect-ratio=1/1 ! "
        # Scale to 224x224 and RGB
        "videoscale ! video/x-raw,format=RGB,width=224,height=224 ! "
        # Send out the resulting frame to an appsink (where we can pick it up from Python)
        "queue max-size-buffers=2 leaky=downstream ! "
        "appsink name=sink drop=true sync=false max-buffers=1 emit-signals=true"
    )
    PREFIX = '[GStreamer]'
    FILE_NAME = 'gstreamer.png'

print('Pipeline:', PIPELINE)

# Create a new GStreamer pipeline, that takes data from DEVICE, and spits out 224x224 RGB arrays
pipeline = Gst.parse_launch(PIPELINE)
appsink = pipeline.get_by_name("sink")

# Keep track when "frame_ready" was sent, so we can calculate the time from the moment the camera
# spat out a frame until when we can do something with the frame in Python
last_frame_ready = -1

def on_frame_ready(element, buffer):
    global last_frame_ready
    last_frame_ready = time.perf_counter()

identity = pipeline.get_by_name("frame_ready")
identity.set_property("signal-handoffs", True)
identity.connect("handoff", on_frame_ready)

# The on_new_sample event is ran on the GStreamer thread; so we want to capture that here,
# and pull the frame from the queue in Python main thread again (so we don't do work on the GStreamer thread)
q = collections.deque(maxlen=2)
lock = threading.Lock()
cv = threading.Condition(lock)

# This takes the image off the appsink, and turns it into a tightly-packed Numpy RGB array
# e.g. IMSDK uses row-stride padding.
def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w = s.get_value("width")
    h = s.get_value("height")

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if ok:
        try:
            fmt = s.get_value("format")  # e.g. "RGB", "BGR", "RGBA", "GRAY8"
            if fmt in ("RGB", "BGR"):
                C = 3
            elif fmt in ("RGBA", "BGRA", "ARGB", "ABGR"):
                C = 4
            elif fmt in ("GRAY8",):
                C = 1
            else:
                # Only single-plane packed formats handled here
                raise RuntimeError(f"Unsupported packed format for NumPy-only path: {fmt}")

            rowbytes = w * C
            # Infer row stride from total mapped size (works for single-plane packed formats)
            rowstride = mapinfo.size // h
            if rowstride * h != mapinfo.size or rowstride < rowbytes:
                raise RuntimeError(f"Unexpected buffer layout: size={mapinfo.size}, h={h}, inferred stride={rowstride}, rowbytes={rowbytes}")

            base = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * rowstride)

            if C == 1:
                # (h, w) grayscale
                arr_view = np.lib.stride_tricks.as_strided(
                    base[: h * rowstride],
                    shape=(h, w),
                    strides=(rowstride, 1),
                )
            else:
                # (h, w, C) packed RGB/BGR/RGBA/...
                arr_view = np.lib.stride_tricks.as_strided(
                    base[: h * rowstride],
                    shape=(h, w, C),
                    strides=(rowstride, C, 1),
                )

            arr = arr_view.copy()  # tight, safe after unmap

            with lock:
                if len(q) == q.maxlen: q.popleft()
                q.append(arr)
                cv.notify()     # notify Python loop
        finally:
            buf.unmap(mapinfo)
    return Gst.FlowReturn.OK

# Start the thread
appsink.connect("new-sample", on_new_sample)
pipeline.set_state(Gst.State.PLAYING)

try:
    # Python processing loop: never blocks on camera
    while True:
        with cv:
            # sleep until at least one frame is present
            cv.wait_for(lambda: len(q) > 0)
            frame = q.pop()          # grab the newest
            q.clear()                # (optional) drop older stale ones

            t_ready = last_frame_ready  # read under same lock

        print(f'{PREFIX} process: {(time.perf_counter() - t_ready)*1000:.2f} ms')

        # You now have "frame" - which is a correctly sized numpy array

        # Save image to disk
        Image.fromarray(frame, "RGB").save("/tmp/" + FILE_NAME)
        os.makedirs('out', exist_ok=True)
        os.rename('/tmp/' + FILE_NAME, 'out/' + FILE_NAME)

finally:
    pipeline.set_state(Gst.State.NULL)
