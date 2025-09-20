import time, threading, collections, numpy as np
import gi, os
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
from PIL import Image

Gst.init(None)

# Use 0 for the large camera, or 1 for the small camera
CAMERA_ID = 0

if CAMERA_ID == 0:
    RESOLUTION = 'width=1920,height=1080'
elif CAMERA_ID == 1:
    RESOLUTION = 'width=1280,height=720'
else:
    print(f'Unknown camera ID: {CAMERA_ID}')
    exit(1)

# GStreamer pipeline using the qtiqmmfsrc plugin
PIPELINE = (
    # Video source
    f"qtiqmmfsrc name=camsrc camera={CAMERA_ID} ! "
    # Properties for the video source
    f"video/x-raw,{RESOLUTION} ! "

    # Convert to RGB
    "qtivtransform ! video/x-raw,format=RGB ! "

    # OPTIONAL... If you want to resize the image within the pipeline, you can remove the line above, and replace with:
    # 1) Crop (square), the crop syntax is ('<X, Y, WIDTH, HEIGHT >') - this is a center crop for camera 0
    # f'qtivtransform crop="<0, 420, 1080, 1080>" ! '
    # 2) then resize to 224x224
    # "video/x-raw,format=RGB,width=224,height=224 ! "

    # Send out the resulting frame to an appsink (where we can pick it up from Python)
    "queue max-size-buffers=2 leaky=downstream ! "
    "appsink name=sink drop=true sync=false max-buffers=1 emit-signals=true"
)
print('Pipeline:', PIPELINE)

# Create a new GStreamer pipeline, that takes data from DEVICE, and spits out 224x224 RGB arrays
pipeline = Gst.parse_launch(PIPELINE)
appsink = pipeline.get_by_name("sink")

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

        # You now have "frame" - which is a correctly sized uint8 RGB numpy array

        # Save image to disk
        Image.fromarray(frame, "RGB").save("/tmp/imsdk.png")
        os.makedirs('out', exist_ok=True)
        os.rename('/tmp/imsdk.png', 'out/imsdk.png')

        print('Saved frame:', frame.shape, 'to out/imsdk.png')

finally:
    pipeline.set_state(Gst.State.NULL)
