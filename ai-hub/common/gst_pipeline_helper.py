from preprocessing import centered_aspect_crop_rect, videocrop_aspect_crop_rect
from gst_helper import has_gst_element
import os

def get_gstreamer_input_pipeline(video_source, video_input_width, video_input_height, resize_mode, interpreter):
    if resize_mode != 'fit-short' and resize_mode != 'squash':
        raise Exception(f'Invalid value for --resize-mode ("{resize_mode}"). Should be "fit-short" or "squash".')

    input_details = interpreter.get_input_details()

    input_h = input_details[0]['shape'][1]
    input_w = input_details[0]['shape'][2]

    resize_crop_pipeline = ''

    if has_gst_element('qtivtransform'):
        # IM SDK
        resize_crop_pipeline = f'qtivtransform ! video/x-raw,format=RGB,width={input_w},height={input_h}'
        if resize_mode == 'fit-short':
            if input_w == video_input_width and input_h == video_input_height:
                pass
            else:
                crop_region = centered_aspect_crop_rect(
                    input_w=input_w,
                    input_h=input_h,
                    cam_w=video_input_width,
                    cam_h=video_input_height
                )
                resize_crop_pipeline = f'qtivtransform crop="{crop_region}" ! video/x-raw,format=RGB,width={input_w},height={input_h}'
    else:
        # Fallback to CPU
        if resize_mode == 'squash':
            resize_crop_pipeline = f'videoconvert ! videoscale ! video/x-raw,format=RGB,width={input_w},height={input_h}'
        else:
            # fit-short
            crop_region = videocrop_aspect_crop_rect(
                input_w=input_w,
                input_h=input_h,
                cam_w=video_input_width,
                cam_h=video_input_height
            )

            resize_crop_pipeline = (
                'videoconvert ! '
                'videoscale ! '
                f'videocrop {crop_region} ! '
                f'videoscale ! '
                f'video/x-raw,format=RGB,width={input_w},height={input_h}'
            )

    return (
        # Video source
        f"{video_source} ! "
        # Properties for the video source
        f"video/x-raw,width={video_input_width},height={video_input_height} ! "
        # An identity element so we can track when a new frame is ready (so we can calc. processing time)
        "identity name=frame_ready_webcam silent=false ! "

        # Resize and crop
        f'{resize_crop_pipeline} ! '
        # Event when the crop/scale are done
        "identity name=transform_done silent=false ! "
        # Send out the resulting frame to an appsink (where we can pick it up from Python)
        "queue max-size-buffers=2 leaky=downstream ! "
        "appsink name=frame drop=true sync=false max-buffers=1 emit-signals=true "
    )

def get_gstreamer_output_pipeline_mp4(out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if has_gst_element('v4l2h264enc'):
        return f"videoconvert ! v4l2h264enc ! h264parse config-interval=-1 ! mp4mux faststart=true ! filesink location={out_file}"
    elif has_gst_element('vtenc_h264'):
        return f"videoconvert ! vtenc_h264 realtime=true allow-frame-reordering=false bitrate=4000000 ! h264parse config-interval=1 ! video/x-h264,stream-format=avc,alignment=au ! mp4mux faststart=true ! filesink location={out_file}"
    else:
        raise Exception('Cannot find either "v4l2h264enc" or "vtenc_h264" GStreamer element')
