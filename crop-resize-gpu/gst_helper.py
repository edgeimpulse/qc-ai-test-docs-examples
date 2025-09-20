import time, collections, queue, numpy as np, gi, os
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from PIL import Image

Gst.init(None)

def gst_grouped_frames(pipeline_str: str, timeout_s: float = 0.100):
    """
    Yield (pts, frames_by_sink, marks) tuples from a GStreamer pipeline string.

    - frames_by_sink: dict of {sink_name: numpy RGB array}
    - marks: dict of {identity_name: timestamp (perf_counter)}
    - Emits one tuple per buffer PTS, grouped across all appsinks.
    - If a sink is late, a timeout ensures progress (partial group).
    """

    pipeline = Gst.parse_launch(pipeline_str)

    # discover identities and appsinks
    identity_names = []
    appsinks = []
    it = pipeline.iterate_elements()
    while True:
        ok, item = it.next()
        if ok != Gst.IteratorResult.OK:
            break
        elem = item
        f = elem.get_factory()
        if f:
            if f.get_name() == "identity":
                identity_names.append(elem.get_name())
            elif f.get_name() == "appsink":
                appsinks.append(elem)

    expected_sinks = {s.get_name() for s in appsinks}

    # shared state
    ledger = collections.OrderedDict()
    LEDGER_MAX = 256
    outq = queue.Queue(maxsize=4)

    def ensure_entry(pts):
        if pts not in ledger:
            ledger[pts] = {"marks": {}, "frames": {}, "t_first": time.perf_counter()}
            while len(ledger) > LEDGER_MAX:
                ledger.popitem(last=False)
        return ledger[pts]

    # identity: record timestamp
    def on_identity_handoff(element, buffer):
        name = element.get_name()
        pts = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else -1
        entry = ensure_entry(pts)
        entry["marks"][name] = time.perf_counter()

    for nm in identity_names:
        elem = pipeline.get_by_name(nm)
        elem.set_property("signal-handoffs", True)
        elem.connect("handoff", on_identity_handoff)

    # appsink: grab frame, store, maybe emit group
    def on_new_sample(sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        pts = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            rowstride = mapinfo.size // h
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * rowstride)
            arr = arr.reshape(h, rowstride // 3, 3)[:, :w, :].copy()
        finally:
            buf.unmap(mapinfo)

        entry = ensure_entry(pts)
        entry["frames"][sink.get_name()] = arr

        if expected_sinks.issubset(entry["frames"].keys()):
            payload = (pts, entry["frames"], entry["marks"])
            ledger.pop(pts, None)
            try:
                outq.put_nowait(payload)
            except queue.Full:
                pass
        return Gst.FlowReturn.OK

    for s in appsinks:
        s.set_property("emit-signals", True)
        s.connect("new-sample", on_new_sample)

    # timeout: flush partials
    def on_timeout(_):
        now = time.perf_counter()
        expired = [
            pts
            for pts, e in ledger.items()
            if (now - e["t_first"]) > timeout_s and e["frames"]
        ]
        for pts in expired:
            e = ledger.pop(pts)
            try:
                outq.put_nowait((pts, e["frames"], e["marks"]))
            except queue.Full:
                pass
        return True

    GLib.timeout_add(int(timeout_s * 500), on_timeout, None)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        while True:
            pts, frames_by_sink, marks = outq.get()
            marks['pipeline_finished'] = time.perf_counter()
            yield frames_by_sink, marks
    finally:
        pipeline.set_state(Gst.State.NULL)

def atomic_save_image(frame, path):
    tmp_file = "/tmp/" + os.path.basename(path)
    Image.fromarray(frame).save(tmp_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.rename(tmp_file, path)

def timing_marks_to_str(marks):
    time_str = []

    # Example timing print
    prev = None
    for identity_name, ts in marks.items():
        if prev:
            time_str.append(f"{prev[0]}→{identity_name}: {(ts - prev[1])*1000:.2f}ms")
        prev = (identity_name, ts)

    if (len(time_str) > 0):
        if (len(marks) >= 2):
            last_ts = list(marks.items())[-1][1]
            first_ts = list(marks.items())[0][1]
            return ', '.join(time_str) + f' (total {(last_ts - first_ts)*1000:.2f}ms)'
        return ', '.join(time_str)
    else:
        return 'N/A'
