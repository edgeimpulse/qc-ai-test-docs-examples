import numpy as np

def face_det_lite_postprocessing(interpreter):
    output_details = interpreter.get_output_details()

    # Grab 3 output tensors and dequantize
    q_output_0 = interpreter.get_tensor(output_details[0]['index'])
    scale_0, zero_point_0 = output_details[0]['quantization']
    hm = ((q_output_0.astype(np.float32) - zero_point_0) * scale_0)[0]

    q_output_1 = interpreter.get_tensor(output_details[1]['index'])
    scale_1, zero_point_1 = output_details[1]['quantization']
    box = ((q_output_1.astype(np.float32) - zero_point_1) * scale_1)[0]

    q_output_2 = interpreter.get_tensor(output_details[2]['index'])
    scale_2, zero_point_2 = output_details[2]['quantization']
    landmark = ((q_output_2.astype(np.float32) - zero_point_2) * scale_2)[0]

    # Taken from https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/utils/bounding_box_processing.py#L369
    def get_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """
        Given two tensors of shape (4,) in xyxy format,
        compute the iou between the two boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        return inter_area / float(boxA_area + boxB_area - inter_area)

    # Taken from https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/models/face_det_lite/utils.py
    class BBox:
        # Bounding Box
        def __init__(
            self,
            label: str,
            xyrb: list[int],
            score: float = 0,
            landmark: list | None = None,
            rotate: bool = False,
        ):
            """
            A bounding box plus landmarks structure to hold the hierarchical result.
            parameters:
                label:str the class label
                xyrb: 4 list for bbox left, top,  right bottom coordinates
                score:the score of the deteciton
                landmark: 10x2 the landmark of the joints [[x1,y1], [x2,y2]...]
            """
            self.label = label
            self.score = score
            self.landmark = landmark
            self.x, self.y, self.r, self.b = xyrb
            self.rotate = rotate

            minx = min(self.x, self.r)
            maxx = max(self.x, self.r)
            miny = min(self.y, self.b)
            maxy = max(self.y, self.b)
            self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

        @property
        def width(self) -> int:
            return self.r - self.x + 1

        @property
        def height(self) -> int:
            return self.b - self.y + 1

        @property
        def box(self) -> list[int]:
            return [self.x, self.y, self.r, self.b]

        @box.setter
        def box(self, newvalue: list[int]) -> None:
            self.x, self.y, self.r, self.b = newvalue

        @property
        def haslandmark(self) -> bool:
            return self.landmark is not None

        @property
        def xywh(self) -> list[int]:
            return [self.x, self.y, self.width, self.height]

    # Taken from https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/models/face_det_lite/utils.py
    def nms(objs: list[BBox], iou: float = 0.5) -> list[BBox]:
        """
        nms function customized to work on the BBox objects list.
        parameter:
            objs: the list of the BBox objects.
        return:
            the rest of the BBox after nms operation.
        """
        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):
            if flags[index] != 0:
                continue

            keep.append(obj)
            for j in range(index + 1, len(objs)):
                # if flags[j] == 0 and obj.iou(objs[j]) > iou:
                if (
                    flags[j] == 0
                    and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
                ):
                    flags[j] = 1
        return keep

    # Ported from https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/models/face_det_lite/utils.py#L110
    # The original code uses torch.Tensor, this uses native numpy arrays
    def detect(
        hm: np.ndarray,           # (H, W, 1), float32
        box: np.ndarray,          # (H, W, 4), float32
        landmark: np.ndarray,     # (H, W, 10), float32
        threshold: float = 0.2,
        nms_iou: float = 0.2,
        stride: int = 8,
    ) -> list:
        def _sigmoid(x: np.ndarray) -> np.ndarray:
            # stable-ish sigmoid
            out = np.empty_like(x, dtype=np.float32)
            np.negative(x, out=out)
            np.exp(out, out=out)
            out += 1.0
            np.divide(1.0, out, out=out)
            return out

        def _maxpool3x3_same(x_hw: np.ndarray) -> np.ndarray:
            """
            x_hw: (H, W) single-channel array.
            3x3 max pool, stride=1, padding=1 (same as PyTorch F.max_pool2d(kernel=3,stride=1,padding=1))
            Pure NumPy using stride tricks.
            """
            H, W = x_hw.shape
            # pad with -inf so edges don't borrow smaller values
            pad = 1
            xpad = np.pad(x_hw, ((pad, pad), (pad, pad)), mode='constant', constant_values=-np.inf)

            # build 3x3 sliding windows using as_strided
            s0, s1 = xpad.strides
            shape = (H, W, 3, 3)
            strides = (s0, s1, s0, s1)
            windows = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=strides, writeable=False)
            # max over the 3x3 window
            return windows.max(axis=(2, 3))

        def _topk_desc(values_flat: np.ndarray, k: int):
            """Return (topk_values_sorted, topk_indices_sorted_desc)."""
            if k <= 0:
                return np.array([], dtype=values_flat.dtype), np.array([], dtype=np.int64)
            k = min(k, values_flat.size)
            # argpartition for top-k by value
            idx_part = np.argpartition(-values_flat, k - 1)[:k]
            # sort those k by value desc
            order = np.argsort(-values_flat[idx_part])
            idx_sorted = idx_part[order]
            return values_flat[idx_sorted], idx_sorted

        # 1) sigmoid heatmap
        hm = _sigmoid(hm.astype(np.float32, copy=False))

        # squeeze channel -> (H, W)
        hm_hw = hm[..., 0]

        # 2) 3x3 max-pool same
        hm_pool = _maxpool3x3_same(hm_hw)

        # 3) local maxima mask (keep equal to pooled)
        # (like (hm == hm_pool).float() * hm in torch)
        keep = (hm_hw >= hm_pool)  # >= to keep plateaus, mirrors torch equality on floats closely enough
        candidate_scores = np.where(keep, hm_hw, 0.0).ravel()

        # 4) topk up to 2000
        num_candidates = int(keep.sum())
        k = min(num_candidates, 2000)
        scores_k, flat_idx_k = _topk_desc(candidate_scores, k)

        H, W = hm_hw.shape
        ys = (flat_idx_k // W).astype(np.int32)
        xs = (flat_idx_k %  W).astype(np.int32)

        # 5) gather boxes/landmarks and build outputs
        objs = []
        for cx, cy, score in zip(xs, ys, scores_k):
            if score < threshold:
                # because scores_k is sorted desc, we can break
                break

            # box offsets at (cy, cx): [x, y, r, b]
            x, y, r, b = box[cy, cx].astype(np.float32, copy=False)

            # convert to absolute xyrb in pixels (same math as torch code)
            cxcycxcy = np.array([cx, cy, cx, cy], dtype=np.float32)
            xyrb = (cxcycxcy + np.array([-x, -y,  r,  b], dtype=np.float32)) * float(stride)
            xyrb = xyrb.astype(np.int32, copy=False).tolist()

            # landmarks: first 5 x, next 5 y
            x5y5 = landmark[cy, cx].astype(np.float32, copy=False)
            x5y5 = x5y5 + np.array([cx]*5 + [cy]*5, dtype=np.float32)
            x5y5 *= float(stride)
            box_landmark = list(zip(x5y5[:5].tolist(), x5y5[5:].tolist()))

            objs.append(BBox("0", xyrb=xyrb, score=float(score), landmark=box_landmark))

        if nms_iou != -1:
            return nms(objs, iou=nms_iou)
        return objs

    # Detection code from https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/models/face_det_lite/app.py#L77
    dets = detect(hm, box, landmark, threshold=0.55, nms_iou=-1, stride=8)
    res = []
    for n in range(0, len(dets)):
        xmin, ymin, w, h = dets[n].xywh
        score = dets[n].score

        L = int(xmin)
        R = int(xmin + w)
        T = int(ymin)
        B = int(ymin + h)
        W = int(w)
        H = int(h)

        if L < 0 or T < 0 or R >= 640 or B >= 480:
            if L < 0:
                L = 0
            if T < 0:
                T = 0
            if R >= 640:
                R = 640 - 1
            if B >= 480:
                B = 480 - 1

        # Enlarge bounding box to cover more face area
        b_Left = L - int(W * 0.05)
        b_Top = T - int(H * 0.05)
        b_Width = int(W * 1.1)
        b_Height = int(H * 1.1)

        if (
            b_Left >= 0
            and b_Top >= 0
            and b_Width - 1 + b_Left < 640
            and b_Height - 1 + b_Top < 480
        ):
            L = b_Left
            T = b_Top
            W = b_Width
            H = b_Height
            R = W - 1 + L
            B = H - 1 + T

        res.append([L, T, W, H, score])

    return res
