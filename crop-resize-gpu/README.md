# Crop and resize: OpenCV vs GStreamer vs IM SDK

This compares cropping and resizing an image from a webcam using OpenCV, vanilla GStreamer and Qualcomm's IM SDK (which runs on GPU).

## Python

To test:

1. Make sure you have the IM SDK installed on your development board.
2. Create a new venv, and install dependencies:

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```

3. Run the example:
    * If you want to use a USB webcam:
        1. Find out the device ID:

            ```bash
            v4l2-ctl --list-devices
            # msm_vidc_media (platform:aa00000.video-codec):
            #         /dev/media0
            #
            # msm_vidc_decoder (platform:msm_vidc_bus):
            #         /dev/video32
            #         /dev/video33
            #
            # C922 Pro Stream Webcam (usb-0000:01:00.0-2):
            #         /dev/video2     <-- So /dev/video2
            #         /dev/video3
            #         /dev/media3
            ```

        2. Run the example:

            **OpenCV**

            ```bash
            python3 opencv.py --device-id 2
            # [OpenCV] 13.46ms.
            # [OpenCV] 15.01ms.
            # [OpenCV] 12.56ms.
            ```

            **GStreamer**

            ```bash
            python3 gstreamer.py --video-source "v4l2src device=/dev/video2"
            # [GStreamer] process: 47.16 ms
            # [GStreamer] process: 26.52 ms
            # [GStreamer] process: 55.05 ms
            ```

            **IM SDK**

            ```bash
            python3 imsdk.py --video-source "v4l2src device=/dev/video2"
            # [IMSDK] process: 9.19 ms
            # [IMSDK] process: 9.12 ms
            # [IMSDK] process: 8.38 ms
            ```

    * If you're on the RB3 Gen 2 Vision Kit, and want to use the built-in camera:

        **OpenCV**

        Won't run.

        **GStreamer**

        ```bash
        python3 gstreamer.py --video-source "qtiqmmfsrc name=camsrc camera=0"
        # Frame ready
        #     Data: name=frame (224, 224, 3)
        #     Timings: frame_ready_webcam→transform_done: 18.48ms, transform_done→pipeline_finished: 2.60ms (total 21.09ms)
        # Frame ready
        #     Data: name=frame (224, 224, 3)
        #     Timings: frame_ready_webcam→transform_done: 18.56ms, transform_done→pipeline_finished: 3.39ms (total 21.95ms)
        ```

        **IM SDK**

        ```bash
        python3 imsdk.py --video-source "qtiqmmfsrc name=camsrc camera=0"
        # Frame ready
        #     Data: name=frame (224, 224, 3)
        #     Timings: frame_ready→transform_done: 2.99ms, transform_done→pipeline_finished: 1.38ms (total 4.37ms)
        # Frame ready
        #     Data: name=frame (224, 224, 3)
        #     Timings: frame_ready→transform_done: 2.05ms, transform_done→pipeline_finished: 2.04ms (total 4.09ms)
        ```
