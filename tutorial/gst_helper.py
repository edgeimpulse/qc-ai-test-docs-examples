import time, collections, queue, numpy as np, gi, os, threading
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from PIL import Image
import urllib.request
from contextlib import contextmanager

Gst.init(None)

def _bucket(ns_value: int, period_ns: int) -> int:
    return int((ns_value + period_ns // 2) // period_ns) * period_ns

def _frame_period_ns_from_pad(pad) -> int:
    caps = pad.get_current_caps()
    if not caps or caps.get_size() == 0:
        return 33333333  # ~30fps
    s = caps.get_structure(0)
    if s and s.has_field("framerate"):
        ok, num, den = s.get_fraction("framerate")
        if num > 0 and den > 0:
            return int(1_000_000_000 * den / num)
    return 33333333

def gst_grouped_frames(pipeline_str: str, element_properties = {}):
    """
    Yield (pts, frames_by_sink, marks) tuples from a GStreamer pipeline string.

    - frames_by_sink: dict of {sink_name: numpy RGB array}
    - marks: dict of {identity_name: timestamp (perf_counter)}
    - Emits one tuple per buffer PTS, grouped across all appsinks.
    - If a sink is late, a timeout ensures progress (partial group).
    """
    timeout_s = 0.100

    if (pipeline_str.strip().startswith('!')):
        raise Exception('First argument of pipeline is empty, did you not set the IMSDK_VIDEO_SOURCE env variable? E.g.:\n' +
            '    export IMSDK_VIDEO_SOURCE="v4l2src device=/dev/video2"')

    pipeline = Gst.parse_launch(pipeline_str)

    # discover identities and appsinks (recursively)
    identity_elems = []
    appsinks = []

    def walk(bin_or_elem):
        if isinstance(bin_or_elem, Gst.Bin):
            it = bin_or_elem.iterate_elements()
            while True:
                ok, item = it.next()
                if ok != Gst.IteratorResult.OK:
                    break
                walk(item)
        else:
            f = bin_or_elem.get_factory()
            if f and f.get_name() == "appsink":
                appsinks.append(bin_or_elem)
            # any element with a 'signal-handoffs' property behaves like identity for handoffs
            if bin_or_elem.find_property("signal-handoffs") is not None:
                identity_elems.append(bin_or_elem)
    walk(pipeline)

    expected_sinks = {s.get_name() for s in appsinks}
    identity_names = [e.get_name() for e in identity_elems]

    # ---- Track segment per PAD
    seg_by_pad = {}          # pad -> Gst.Segment
    seg_lock = threading.Lock()

    def set_segment_for_pad(pad, event):
        seg = event.parse_segment()
        with seg_lock:
            seg_by_pad[pad] = seg

    def to_running_time_ns(pad, pts):
        if pts == Gst.CLOCK_TIME_NONE:
            return None
        with seg_lock:
            seg = seg_by_pad.get(pad)
        if not seg:
            return None
        rt = seg.to_stream_time(Gst.Format.TIME, pts)
        # GI returns a Python int (nanoseconds) or Gst.CLOCK_TIME_NONE (-1) if invalid
        if rt == Gst.CLOCK_TIME_NONE or rt < 0:
            return None
        return int(rt)

    # shared state
    ledger = collections.OrderedDict()
    LEDGER_MAX = 256
    outq = queue.Queue(maxsize=4)
    frame_index = 0

    def ensure_entry(pts):
        if pts not in ledger:
            ledger[pts] = {"marks": {}, "frames": {}, "t_first": time.perf_counter()}
            while len(ledger) > LEDGER_MAX:
                ledger.popitem(last=False)
        return ledger[pts]

    # identity: record timestamp
    def on_identity_handoff(element, buf):
        # nonlocal frame_index
        name = element.get_name()
        pts = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1

        pad = element.get_static_pad("src")
        rt_ns = to_running_time_ns(pad, buf.pts)
        if rt_ns is None:
            key = buf.pts
        else:
            key = _bucket(rt_ns, frame_period_ns)

        entry = ensure_entry(key)
        entry["marks"][name] = time.perf_counter()

    for elem in identity_elems:
        elem.set_property("signal-handoffs", True)
        elem.connect("handoff", on_identity_handoff)

    # appsink: grab frame, store, maybe emit group
    def on_new_sample(sink):
        # nonlocal frame_index
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        fmt = caps.get_value("format")  # e.g. "RGB", "BGR", "RGBA", "BGRA", "GRAY8"
        pts = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1

        name = sink.get_name()
        pad = sink.get_static_pad("sink")
        rt_ns = to_running_time_ns(pad, buf.pts)
        if rt_ns is None:
            key = buf.pts
        else:
            key = _bucket(rt_ns, frame_period_ns)

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            if (w is not None and h is not None and fmt is not None):
                if fmt in ("RGB", "BGR"):
                    C = 3
                elif fmt in ("RGBA", "BGRA", "ARGB", "ABGR"):
                    C = 4
                elif fmt in ("GRAY8",):
                    C = 1
                else:
                    raise RuntimeError(f"Unsupported format: {fmt}")

                rowstride = mapinfo.size // h
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * rowstride)
                arr = arr.reshape(h, rowstride // C, C)[:, :w, :].copy()

                # Fix channel ordering to RGB / RGBA
                if fmt == "BGR":
                    arr = arr[:, :, ::-1]  # BGR → RGB
                elif fmt == "BGRA":
                    arr = arr[:, :, [2, 1, 0, 3]]  # BGRA → RGBA
                elif fmt == "ARGB":  # ARGB → RGBA
                    arr = arr[:, :, [1, 2, 3, 0]]
                elif fmt == "ABGR":  # ABGR → RGBA
                    arr = arr[:, :, [3, 2, 1, 0]]
                # (RGB and RGBA are already fine)
            else:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(mapinfo)

        entry = ensure_entry(key)
        entry["frames"][name] = arr

        if expected_sinks.issubset(entry["frames"].keys()):
            entry['marks']['pipeline_finished'] = time.perf_counter()
            payload = (pts, entry["frames"], entry["marks"])
            # frame_index = frame_index + 1
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
                e['marks']['pipeline_finished'] = time.perf_counter()
                outq.put_nowait((pts, e["frames"], e["marks"]))
            except queue.Full:
                pass
        return True

    def get_latest(q):
        item = q.get()            # will block until *one* item is available
        while True:
            try:
                item = q.get_nowait()  # grab newer ones if they arrived
            except queue.Empty:
                return item

    GLib.timeout_add(int(timeout_s * 500), on_timeout, None)

    # Override properties (from element_properties)
    for el_key in element_properties.keys():
        el = pipeline.get_by_name(el_key)
        for prop in element_properties[el_key].keys():
            value = element_properties[el_key][prop]
            el.set_property(prop, value)
            if el.get_property(prop) is None:
                raise Exception(f'Failed to set property "{prop}" on element "{el_key}" (value is None after setting)')

    # ---- Probes
    # For each identity: attach a probe on its SRC pad to get SEGMENT + BUFFERS
    def attach_identity_probe(identity):
        pad = identity.get_static_pad("src")
        name = identity.get_name()

        def cb(pad, info, _user):
            typ = info.type
            # Track segment events
            if typ & Gst.PadProbeType.EVENT_DOWNSTREAM:
                ev = info.get_event()
                if ev.type == Gst.EventType.SEGMENT:
                    set_segment_for_pad(pad, ev)
                return Gst.PadProbeReturn.OK

            return Gst.PadProbeReturn.OK

        pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM | Gst.PadProbeType.BUFFER,
            cb, None
        )

    # For each appsink: attach probe on its SINK pad to see buffers before the sink consumes them
    def attach_appsink_probe(appsink):
        sink_pad = appsink.get_static_pad("sink")
        name = appsink.get_name()

        def cb(pad, info, _user):
            typ = info.type
            if typ & Gst.PadProbeType.EVENT_DOWNSTREAM:
                ev = info.get_event()
                if ev.type == Gst.EventType.SEGMENT:
                    set_segment_for_pad(pad, ev)
                return Gst.PadProbeReturn.OK

            return Gst.PadProbeReturn.OK

        sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM | Gst.PadProbeType.BUFFER,
            cb, None
        )

    # Attach probes
    for iden in identity_elems:
        attach_identity_probe(iden)
    for sink in appsinks:
        attach_appsink_probe(sink)

    pipeline.set_state(Gst.State.PLAYING)

    all_frame_periods = []
    frame_period_ns = -1

    for appsink in appsinks:
        appsink_sink_pad = appsink.get_static_pad("sink")
        frame_period_ns = _frame_period_ns_from_pad(appsink_sink_pad)
        all_frame_periods.append(frame_period_ns)

    if not len(set(all_frame_periods)) <= 1:
        raise Exception('appsinks have different values for _frame_period_ns_from_pad... cannot handle this: values=[' +
            ','.join(str(x) for x in all_frame_periods) + ']')

    try:
        while True:
            pts, frames_by_sink, marks = get_latest(outq)
            yield frames_by_sink, marks
    finally:
        pipeline.set_state(Gst.State.NULL)

def atomic_save_image(frame, path):
    tmp_file = "/tmp/" + os.path.basename(path)
    Image.fromarray(frame).save(tmp_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.rename(tmp_file, path)

def atomic_save_pillow_image(img, path):
    tmp_file = "/tmp/" + os.path.basename(path)
    img.save(tmp_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.rename(tmp_file, path)

def atomic_save_numpy_buffer(buf, path):
    tmp_file = "/tmp/" + os.path.basename(path)
    buf.tofile(tmp_file)
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

def download_file_if_needed(model_path, model_url):
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_url = model_url
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloading {model_path} OK")
    return model_path

def softmax(x, axis=-1):
    # subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

@contextmanager
def mark_performance(name, marks):
    start = time.perf_counter()

    try:
        yield  # run the block inside `with`
    finally:
        end = time.perf_counter()
        # Add an extra mark, so we have timing info for the complete pipeline
        marks[name] = list(marks.items())[-1][1] + (end - start)
