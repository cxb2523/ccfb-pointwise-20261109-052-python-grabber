# Captures a frame from the camera every second and save it to a file.
# Demonstrates switching the grayscale filter on and off while the graph keeps
# running: no stop/rebuild, the next delivered frame uses the new mode.

import cv2
from datetime              import datetime
from threading             import Event, Thread
from pygrabber.dshow_graph import FilterGraph
from os                    import path

PAUSE_BETWEEN_CAPTURE = 1  # 1 sec.
OUTPUT_FOLDER         = r"C:\test\\"
CAMERA_INDEX          = 0 # Using first camera found
GRAYSCALE_ON_AFTER    = 3 # switch to grayscale after this many frames
GRAYSCALE_OFF_AFTER   = 6 # switch back to color after this many frames

global start_time


def capture_photos_loop(event_):
    global start_time
    start_time = datetime.now()
    frame_count = 0
    while not event_.isSet():
        # Toggle the filter while streaming: the change applies to the next
        # frame the SampleGrabber callback delivers, without touching the graph.
        if frame_count == GRAYSCALE_ON_AFTER:
            print("Switching grayscale ON")
            graph.set_grayscale(True)
        elif frame_count == GRAYSCALE_OFF_AFTER:
            print("Switching grayscale OFF")
            graph.set_grayscale(False)
        graph.grab_frame()
        frame_count += 1
        event_.wait(PAUSE_BETWEEN_CAPTURE)


def show_image(image):
    global start_time
    capture_time = datetime.now() - start_time
    image_file_name = path.join(OUTPUT_FOLDER, str(capture_time.seconds * 1000 + int(capture_time.microseconds / 1000)) + ".jpg")
    cv2.imwrite(image_file_name, image)
    print(f"Image {image_file_name} written (shape={image.shape}, grayscale={graph.is_grayscale_enabled()})")


if __name__ == "__main__":
    event          = Event()
    capture_thread = Thread(target=capture_photos_loop, args=(event, ))
    graph          = FilterGraph()
    devices        = graph.get_input_devices()

    print(f"Connecting to device {devices[CAMERA_INDEX]}")
    graph.add_video_input_device(CAMERA_INDEX)
    graph.add_sample_grabber(lambda image: show_image(image))
    graph.add_null_render()
    graph.prepare_preview_graph()
    graph.run()

    capture_thread.start()
    input(f"Capturing images every {PAUSE_BETWEEN_CAPTURE}s, press ENTER to terminate.")
    event.set()
    capture_thread.join()
    print("Done")
