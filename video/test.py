import ffmpeg
import os
import cv2

root_path = os.getcwd()
video_root_path = f"{root_path}/data/video"
image_root_path = f"{root_path}/data/image"


def get_video_path(video_id, ratio):
    return f"{video_root_path}/{video_id}/{ratio}/{video_id}.mp4"


def get_image_path(video_id, ratio):
    if not os.path.exists(f"{image_root_path}/{video_id}"):
        os.mkdir(f"{image_root_path}/{video_id}")
    elif not os.path.exists(f"{image_root_path}/{video_id}/{ratio}"):
        os.mkdir(f"{image_root_path}/{video_id}/{ratio}")
    return f"{image_root_path}/{video_id}/{ratio}"


def get_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream
         for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None)
    return video_stream


def video_to_image(video_path, image_path):
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    count = 0
    success = True
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    while success:
        count += 1
        cv2.imwrite(f"{image_path}/{count}.jpg", image)
        success, image = cap.read()
        print('Read a new frame: ', count)
    return


if __name__ == "__main__":
    video_path = get_video_path("7150557724969831681", "720p")
    video_info = get_video_info(video_path)
    image_path = get_image_path("7150557724969831681", "720p")
    video_to_image(video_path, image_path)