# Get data from the built-in camera on the RB3 Gen 2 Vision Kit

The [Qualcomm Dragonwing RB3 Gen 2 Development Kit](https://qc-ai-test.gitbook.io/qc-ai-test-docs/device-setup/rb3-gen2-vision-kit) has two built-in cameras, but you can only read data from the cameras using a GStreamer pipeline using the `qtiqmmfsrc` plugin (part of the IM SDK).

## Python

To get data from the built-in cameras into a Python script:

1. Ensure your RB3 Gen 2 Development Kit runs Ubuntu 24.
2. Install dependencies:

    ```bash
    # Add the Qualcomm IoT PPA
    sudo apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa

    # Install GStreamer and the IM SDK
    sudo apt update
    sudo apt install -y gstreamer1.0-tools tensorflow-lite-qcom-apps gstreamer1.0-qcom-sample-apps gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps gstreamer1.0-plugins-qcom-good
    ```

3. Create a new venv, and install dependencies:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```

4. Run the example:

    ```bash
    python3 builtin_camera_to_python.py
    ```

5. Images from the camera are stored in the `out/` directory.

## Camera -> files

To stream camera to data to a file (e.g. to use it in another environment), open a terminal on your device and run:

**Large camera**

```bash
mkdir -p out
gst-launch-1.0 qtiqmmfsrc name=camsrc camera=0 ! video/x-raw,width=1920,height=1080 ! jpegenc ! multifilesink location=out/frame%05d.jpg
```

**Small camera**

```bash
mkdir -p out
gst-launch-1.0 qtiqmmfsrc name=camsrc camera=1 ! video/x-raw,width=1280,height=720 ! jpegenc ! multifilesink location=out/frame%05d.jpg
```

This writes frames to the `out/` directory.
