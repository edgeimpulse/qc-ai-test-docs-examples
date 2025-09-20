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
            python3 gstreamer.py --video-source "v4l2src device=/dev/video2" --use-imsdk
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
        # [GStreamer] process: 19.69 ms
        # [GStreamer] process: 19.35 ms
        # [GStreamer] process: 20.28 ms
        ```

        **IM SDK**

        ```bash
        python3 gstreamer.py --video-source "v4l2src device=/dev/video2" --use-imsdk
        # [IMSDK] process: 3.79 ms
        # [IMSDK] process: 3.71 ms
        # [IMSDK] process: 5.37 ms
        ```
