import numpy as np

POSENET_PART_NAMES = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

def face_det_lite_postprocessing(interpreter):
    # Grab 3 output tensors and (optionally) dequantize
    hm = read_output_tensor(interpreter, 0)
    box = read_output_tensor(interpreter, 0)
    landmark = read_output_tensor(interpreter, 0)

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

def posenet_postprocessing(interpreter):
    # Grab 5 output tensors and (optionally) dequantize
    heatmaps_result = read_output_tensor(interpreter, 0)
    offsets_result = read_output_tensor(interpreter, 1)
    displacement_fwd_result = read_output_tensor(interpreter, 2)
    displacement_bwd_result = read_output_tensor(interpreter, 3)
    max_vals = read_output_tensor(interpreter, 4)

    # Code from qai_hub_models/models/posenet_mobilenet/app.py

    NUM_KEYPOINTS = len(POSENET_PART_NAMES)

    PART_IDS = {pn: pid for pid, pn in enumerate(POSENET_PART_NAMES)}
    LOCAL_MAXIMUM_RADIUS = 1

    POSE_CHAIN = [
        ("nose", "leftEye"),
        ("leftEye", "leftEar"),
        ("nose", "rightEye"),
        ("rightEye", "rightEar"),
        ("nose", "leftShoulder"),
        ("leftShoulder", "leftElbow"),
        ("leftElbow", "leftWrist"),
        ("leftShoulder", "leftHip"),
        ("leftHip", "leftKnee"),
        ("leftKnee", "leftAnkle"),
        ("nose", "rightShoulder"),
        ("rightShoulder", "rightElbow"),
        ("rightElbow", "rightWrist"),
        ("rightShoulder", "rightHip"),
        ("rightHip", "rightKnee"),
        ("rightKnee", "rightAnkle"),
    ]

    PARENT_CHILD_TUPLES = [
        (PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN
    ]
    CONNECTED_PART_NAMES = [
        ("leftHip", "leftShoulder"),
        ("leftElbow", "leftShoulder"),
        ("leftElbow", "leftWrist"),
        ("leftHip", "leftKnee"),
        ("leftKnee", "leftAnkle"),
        ("rightHip", "rightShoulder"),
        ("rightElbow", "rightShoulder"),
        ("rightElbow", "rightWrist"),
        ("rightHip", "rightKnee"),
        ("rightKnee", "rightAnkle"),
        ("leftShoulder", "rightShoulder"),
        ("leftHip", "rightHip"),
    ]

    CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]
    OUTPUT_STRIDE = 16


    def traverse_to_targ_keypoint(
        edge_id: int,
        source_keypoint: np.ndarray,
        target_keypoint_id: int,
        scores: np.ndarray,
        offsets: np.ndarray,
        displacements: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Given a source keypoint and target_keypoint_id,
            predict the score and coordinates of the target keypoint.

        Parameters
        ----------
            edge_id: Index of the edge being considered.
                Equivalent to the index in `POSE_CHAIN`.
            source_keypoint: (y, x) coordinates of the keypoint.
            target_keypoint_id: Which body part type of the 17 this keypoint is.
            scores: See `decode_multiple_poses`.
            offsets: See `decode_multiple_poses`.
            displacements: See `decode_multiple_poses`.

        Returns
        -------
            Tuple of target keypoint score and coordinates.
        """
        height = scores.shape[1]
        width = scores.shape[2]

        source_keypoint_indices = np.clip(
            np.round(source_keypoint / OUTPUT_STRIDE),
            a_min=0,
            a_max=[height - 1, width - 1],
        ).astype(np.int32)

        displaced_point = (
            source_keypoint
            + displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]
        )

        displaced_point_indices = np.clip(
            np.round(displaced_point / OUTPUT_STRIDE),
            a_min=0,
            a_max=[height - 1, width - 1],
        ).astype(np.int32)

        score = scores[
            target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
        ]

        image_coord = (
            displaced_point_indices * OUTPUT_STRIDE
            + offsets[
                target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
            ]
        )

        return score, image_coord


    def decode_pose(
        root_score: float,
        root_id: int,
        root_image_coord: np.ndarray,
        scores: np.ndarray,
        offsets: np.ndarray,
        displacements_fwd: np.ndarray,
        displacements_bwd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get all keypoint predictions for a pose given a root keypoint with a high score.

        Parameters
        ----------
            root_score: The confidence score of the root keypoint.
            root_id: Which body part type of the 17 this keypoint is.
            root_image_coord: (y, x) coordinates of the keypoint.
            scores: See `decode_multiple_poses`.
            offsets: See `decode_multiple_poses`.
            displacements_fwd: See `decode_multiple_poses`.
            displacements_bwd: See `decode_multiple_poses`.

        Returns
        -------
            Tuple of list of keypoint scores and list of coordinates.
        """
        num_parts = scores.shape[0]
        num_edges = len(PARENT_CHILD_TUPLES)

        instance_keypoint_scores = np.zeros(num_parts)
        instance_keypoint_coords = np.zeros((num_parts, 2))
        instance_keypoint_scores[root_id] = root_score
        instance_keypoint_coords[root_id] = root_image_coord

        for edge in reversed(range(num_edges)):
            target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
            if (
                instance_keypoint_scores[source_keypoint_id] > 0.0
                and instance_keypoint_scores[target_keypoint_id] == 0.0
            ):
                score, coords = traverse_to_targ_keypoint(
                    edge,
                    instance_keypoint_coords[source_keypoint_id],
                    target_keypoint_id,
                    scores,
                    offsets,
                    displacements_bwd,
                )
                instance_keypoint_scores[target_keypoint_id] = score
                instance_keypoint_coords[target_keypoint_id] = coords

        for edge in range(num_edges):
            source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
            if (
                instance_keypoint_scores[source_keypoint_id] > 0.0
                and instance_keypoint_scores[target_keypoint_id] == 0.0
            ):
                score, coords = traverse_to_targ_keypoint(
                    edge,
                    instance_keypoint_coords[source_keypoint_id],
                    target_keypoint_id,
                    scores,
                    offsets,
                    displacements_fwd,
                )
                instance_keypoint_scores[target_keypoint_id] = score
                instance_keypoint_coords[target_keypoint_id] = coords

        return instance_keypoint_scores, instance_keypoint_coords


    def within_nms_radius_fast(
        pose_coords: np.ndarray, nms_radius: float, point: np.ndarray
    ) -> bool:
        """
        Whether the candidate point is nearby any existing point in `pose_coords`.

        pose_coords:
            Numpy array of points, shape (N, 2).
        nms_radius:
            The distance between two points for them to be considered nearby.
        point:
            The candidate point, shape (2,).
        """
        if not pose_coords.shape[0]:
            return False
        return bool(np.any(np.sum((pose_coords - point) ** 2, axis=1) <= nms_radius**2))


    def get_instance_score_fast(
        exist_pose_coords: np.ndarray,
        nms_radius: int,
        keypoint_scores: np.ndarray,
        keypoint_coords: np.ndarray,
    ) -> float:
        """
        Compute a probability that the given pose is real.
        Equal to the average confidence of each keypoint, excluding keypoints
        that are shared with existing poses.

        Parameters
        ----------
            exist_pose_coords: Keypoint coordinates of poses that have already been found.
                Shape (N, 17, 2)
            nms_radius:
                If two candidate keypoints for the same body part are within this distance,
                    they are considered the same, and the lower confidence one discarded.
            keypoint_scores:
                Keypoint scores for the new pose. Shape (17,)
            keypoint_coords:
                Coordinates for the new pose. Shape (17, 2)

        Returns
        -------
            Confidence score for the pose.
        """
        if exist_pose_coords.shape[0]:
            s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > nms_radius**2
            not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
        else:
            not_overlapped_scores = np.sum(keypoint_scores)
        return not_overlapped_scores / len(keypoint_scores)


    def build_part_with_score_torch(
        score_threshold: float, max_vals: np.array, scores: np.array
    ) -> tuple[np.array, np.array]:
        """
        Get candidate keypoints to be considered the root for a pose.
        Score for the keypoint must be >= all neighboring scores.
        Score must also be above given score_threshold.

        Parameters
        ----------
            score_threshold: Minimum score for a keypoint to be considered as a root.
            max_vals: See `decode_multiple_poses`.
            scores: See `decode_multiple_poses`.

        Returns
        -------
            Tuple of:
                - Torch scores for each keypoint to be considered.
                - Indices of the considered keypoints. Shape (N, 3) where the 3 indices
                    map to the dimensions of the scores tensor with shape (17, h, w).
        """
        max_loc = (scores == max_vals) & (scores >= score_threshold)
        max_loc_idx = np.argwhere(max_loc)
        scores_vec = scores[max_loc]
        sort_idx = np.argsort(-scores_vec)

        return scores_vec[sort_idx], max_loc_idx[sort_idx]


    def decode_multiple_poses(
        scores: np.array,
        offsets: np.array,
        displacements_fwd: np.array,
        displacements_bwd: np.array,
        max_vals: np.array,
        max_pose_detections: int = 10,
        score_threshold: float = 0.25,
        nms_radius: int = 20,
        min_pose_score: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts raw model outputs into image with keypoints drawn.
        Can detect multiple poses in the same image, up to `max_pose_detections`.
        This model has 17 candidate keypoints it predicts.
        In this docstring, (h, w) correspond to height and width of the grid
        and are roughly equal to input image size divided by 16.

        Parameters
        ----------
            scores:
                Tensor of scores in range [0, 1] indicating probability
                    a candidate pose is real. Shape [17, h, w].
            offsets:
                Tensor of offsets for a given keypoint, relative to the grid point.
                    Shape [34, h, w].
            displacements_fwd:
                When tracing the points for a pose, given a source keypoint, this value
                    gives the displacement to the next keypoint in the pose. There are 16
                    connections from one keypoint to another (it's a minimum spanning tree).
                    Shape [32, h, w].
            displacements_bwd:
                Same as displacements_fwd, except when traversing keypoint connections
                    in the opposite direction.
            max_vals:
                Same as scores except with a max pool applied with kernel size 3.
            max_pose_detections:
                Maximum number of distinct poses to detect in a single image.
            score_threshold:
                Minimum score for a keypoint to be considered the root for a pose.
            nms_radius:
                If two candidate keypoints for the same body part are within this distance,
                    they are considered the same, and the lower confidence one discarded.
            min_pose_score:
                Minimum confidence that a pose exists for it to be displayed.

        Returns
        -------
            Tuple of:
                - Numpy array of pose confidence scores.
                - Numpy array of keypoint confidence scores.
                - Numpy array of keypoint coordinates.
        """
        part_scores_pt, part_idx_pt = build_part_with_score_torch(
            score_threshold, max_vals, scores
        )
        part_scores = part_scores_pt
        part_idx = part_idx_pt

        scores_np = scores
        height = scores_np.shape[1]
        width = scores_np.shape[2]
        # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
        offsets_np = (
            offsets.reshape(2, -1, height, width).transpose((1, 2, 3, 0))
        )
        displacements_fwd_np = (
            displacements_fwd
            .reshape(2, -1, height, width)
            .transpose((1, 2, 3, 0))
        )
        displacements_bwd_np = (
            displacements_bwd
            .reshape(2, -1, height, width)
            .transpose((1, 2, 3, 0))
        )

        pose_count = 0
        pose_scores = np.zeros(max_pose_detections)
        pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
        pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

        for root_score, (root_id, root_coord_y, root_coord_x) in zip(
            part_scores, part_idx, strict=False
        ):
            root_coord = np.array([root_coord_y, root_coord_x])
            root_image_coords = (
                root_coord * OUTPUT_STRIDE + offsets_np[root_id, root_coord_y, root_coord_x]
            )

            if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :],
                nms_radius,
                root_image_coords,
            ):
                continue

            keypoint_scores, keypoint_coords = decode_pose(
                root_score,
                root_id,
                root_image_coords,
                scores_np,
                offsets_np,
                displacements_fwd_np,
                displacements_bwd=displacements_bwd_np,
            )

            pose_score = get_instance_score_fast(
                pose_keypoint_coords[:pose_count, :, :],
                nms_radius,
                keypoint_scores,
                keypoint_coords,
            )

            # NOTE this isn't in the original implementation, but it appears that by initially ordering by
            # part scores, and having a max # of detections, we can end up populating the returned poses with
            # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
            # Set min_pose_score to 0. to revert to original behaviour
            if min_pose_score == 0.0 or pose_score >= min_pose_score:
                pose_scores[pose_count] = pose_score
                pose_keypoint_scores[pose_count, :] = keypoint_scores
                pose_keypoint_coords[pose_count, :, :] = keypoint_coords
                pose_count += 1

            if pose_count >= max_pose_detections:
                break

        return pose_scores, pose_keypoint_scores, pose_keypoint_coords

    pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
        heatmaps_result,
        offsets_result,
        displacement_fwd_result,
        displacement_bwd_result,
        max_vals,
        max_pose_detections=10,
        min_pose_score=0.25,
    )

    return pose_scores, keypoint_scores, keypoint_coords

def posenet_draw_skel_and_kp(
    img: np.ndarray,
    instance_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
    min_pose_score: float = 0.5,
    min_part_score: float = 0.5,
) -> None:
    """
    Draw the keypoints and edges on the input numpy array image in-place.

    Parameters
    ----------
        img: Numpy array of the image.
        instance_scores: Numpy array of confidence for each pose.
        keypoint_scores: Numpy array of confidence for each keypoint.
        keypoint_coords: Numpy array of coordinates for each keypoint.
        min_pose_score: Minimum score for a pose to be displayed.
        min_part_score: Minimum score for a keypoint to be displayed.
    """
    import cv2


    PART_IDS = {pn: pid for pid, pn in enumerate(POSENET_PART_NAMES)}

    CONNECTED_PART_NAMES = [
        ("leftHip", "leftShoulder"),
        ("leftElbow", "leftShoulder"),
        ("leftElbow", "leftWrist"),
        ("leftHip", "leftKnee"),
        ("leftKnee", "leftAnkle"),
        ("rightHip", "rightShoulder"),
        ("rightElbow", "rightShoulder"),
        ("rightElbow", "rightWrist"),
        ("rightHip", "rightKnee"),
        ("rightKnee", "rightAnkle"),
        ("leftShoulder", "rightShoulder"),
        ("leftHip", "rightHip"),
    ]

    CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

    def get_adjacent_keypoints(
        keypoint_scores: np.ndarray, keypoint_coords: np.ndarray, score_threshold: float
    ) -> list[np.ndarray]:
        """
        Compute which keypoints should be connected in the image.

        keypoint_scores:
            Scores for all candidate keypoints in the pose.
        keypoint_coords:
            Coordinates for all candidate keypoints in the pose.
        score_threshold:
            If either keypoint in a candidate edge is below this threshold, omit the edge.

        Returns
        -------
            List of (2, 2) numpy arrays containing coordinates of edge endpoints.
        """
        results = []
        for left, right in CONNECTED_PART_INDICES:
            if (
                keypoint_scores[left] < score_threshold
                or keypoint_scores[right] < score_threshold
            ):
                continue
            results.append(
                np.array(
                    [keypoint_coords[left][::-1], keypoint_coords[right][::-1]]
                ).astype(np.int32),
            )
        return results

    adjacent_keypoints = []
    points = []
    sizes = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_connections = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score
        )
        adjacent_keypoints.extend(new_connections)

        for ks, kc in zip(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], strict=False
        ):
            if ks < min_part_score:
                continue
            points.append([kc[1], kc[0]])
            sizes.append(10.0 * ks)

    if points:
        points_np = np.array(points)
        draw_points(img, points_np, color=(255, 255, 0), size=sizes)
        cv2.polylines(img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

def posenet_to_object(pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.25, min_part_score=0.1):
    ret = []

    for i in range(0, len(pose_scores)):
        if pose_scores[i] < min_pose_score: continue

        person = { 'score': pose_scores[i], 'keypoints': [] }
        for j in range(0, len(keypoint_scores[i])):
            keypoint_score = keypoint_scores[i][j]
            if keypoint_score < min_part_score: continue

            keypoint_x = keypoint_coords[i][j][0]
            keypoint_y = keypoint_coords[i][j][1]
            keypoint_label = POSENET_PART_NAMES[j]
            person['keypoints'].append({
                'score': keypoint_score,
                'label': keypoint_label,
                'x': keypoint_x,
                'y': keypoint_y,
            })
        ret.append(person)

    return ret

def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int | list[int] = 10,
):
    """
    Draw the given points on the frame.

    Parameters
    ----------
        frame: np.ndarray
            np array (H W C x uint8, RGB)

        points: np.ndarray | torch.Tensor
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        color: tuple[int, int, int]
            Color of drawn points (RGB)

        size: int
            Size of drawn points

    Returns
    -------
        None; modifies frame in place.
    """
    import cv2

    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    assert isinstance(size, int) or len(size) == len(points)
    cv_keypoints = []
    for i, (x, y) in enumerate(points):
        curr_size = size if isinstance(size, int) else size[i]
        cv_keypoints.append(cv2.KeyPoint(int(x), int(y), curr_size))

    cv2.drawKeypoints(
        frame,
        cv_keypoints,
        outImage=frame,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

def read_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()

    q_output_0 = interpreter.get_tensor(output_details[index]['index'])
    scale_0, zero_point_0 = output_details[index]['quantization']
    if scale_0 == 0.0:
        return q_output_0[0]
    else:
        return ((q_output_0.astype(np.float32) - zero_point_0) * scale_0)[0]
