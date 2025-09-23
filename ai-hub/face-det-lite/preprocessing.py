import numpy as np

def rgb_numpy_arr_to_input_tensor(interpreter, arr, single_channel_behavior: str = 'grayscale'):
    d = interpreter.get_input_details()[0]
    shape = [int(x) for x in d["shape"]]  # e.g. [1, H, W, C] or [1, C, H, W]
    dtype = d["dtype"]
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

    # HWC -> correct layout
    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)

    # Scale 0..1 (all AI Hub image models use this)
    arr = (arr / 255.0).astype(np.float32)

    # Quantize if needed
    if scale and float(scale) != 0.0:
        q = np.rint(arr / float(scale) + int(zp))
        if dtype == np.uint8:
            arr = np.clip(q, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(q, -128, 127).astype(np.int8)

    return np.expand_dims(arr, 0)  # add batch
