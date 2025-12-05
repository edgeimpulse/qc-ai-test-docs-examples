import numpy as np
from PIL import Image

def rgb_numpy_arr_to_input_tensor(interpreter, arr, single_channel_behavior: str = 'grayscale'):
    d = interpreter.get_input_details()[0]
    shape = [int(x) for x in d["shape"]]  # e.g. [1, H, W, C] or [1, C, H, W]
    input_tensor_dtype = d["dtype"]
    scale, zp = d.get("quantization", (0.0, 0))

    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Unexpected input shape: {shape}")

    # Detect layout
    if shape[1] in (1, 3):   # [1, C, H, W]
        layout, C, H, W = "NCHW", shape[1], shape[2], shape[3]
    elif shape[3] in (1, 3): # [1, H, W, C]
        layout, C, H, W = "NHWC", shape[3], shape[1], shape[2]
    else:
        raise ValueError(f"Cannot infer layout from shape {shape}")

    # Load & resize
    if C == 1:
        if single_channel_behavior == 'grayscale':
            # Convert to luminance (H, W)
            gray = np.asarray(Image.fromarray(arr).convert('L'))
        elif single_channel_behavior in ('red', 'green', 'blue'):
            ch_idx = {'red': 0, 'green': 1, 'blue': 2}[single_channel_behavior]
            gray = arr[:, :, ch_idx]
        else:
            raise ValueError(f"Invalid single_channel_behavior: {single_channel_behavior}")
        # Keep shape as HWC with C=1
        arr = gray[..., np.newaxis]

    # image is HWC, but model requires NCHW? -> correct layout
    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)

    # Scale 0..1 (all AI Hub image models use this), your model might use different scaling behavior
    arr = arr.astype(np.float32)
    arr *= 1.0 / 255.0

    # Quantize if needed
    if scale and float(scale) != 0.0:
        inv = np.float32(1.0 / scale)

        np.multiply(arr, inv, out=arr)            # arr = arr * (1/scale)
        if zp:
            np.add(arr, np.float32(zp), out=arr)  # arr += zp
        np.rint(arr, out=arr)                     # round to nearest (ties to even)

        if input_tensor_dtype == np.uint8:        # cast to int8/uint8 depending on reqs
            arr = np.clip(arr, 0, 255).astype(input_tensor_dtype)
        else:
            arr = np.clip(arr, -128, 127).astype(input_tensor_dtype)

    return np.expand_dims(arr, 0)  # add batch

def centered_aspect_crop_rect(input_w: int, input_h: int,
                              cam_w: int, cam_h: int,
                              even: bool = True) -> str:
    """
    Compute a centered crop rectangle so that (crop_w / crop_h) == (input_w / input_h).
    Returns a string like "<x, y, w, h>" for qtivtransform's `crop` property.

    Set even=True to force even x/y/w/h (some hardware prefers even dims).
    """
    assert input_w > 0 and input_h > 0 and cam_w > 0 and cam_h > 0
    target_ar = input_w / input_h
    cam_ar = cam_w / cam_h

    if cam_ar > target_ar:
        # Too wide -> crop width
        crop_h = cam_h
        crop_w = int(round(crop_h * target_ar))
        x = (cam_w - crop_w) // 2
        y = 0
    else:
        # Too tall or equal -> crop height
        crop_w = cam_w
        crop_h = int(round(crop_w / target_ar))
        x = 0
        y = (cam_h - crop_h) // 2

    if even:
        # Make everything even (helps with YUV/hardware paths)
        x &= ~1
        y &= ~1
        crop_w &= ~1
        crop_h &= ~1

    return f"<{x}, {y}, {crop_w}, {crop_h}>"
