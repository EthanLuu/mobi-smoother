import ffmpeg
import os
import cv2
import time

root_path = os.getcwd()
video_root_path = f"{root_path}/data/video"
merged_video_root_path = f"{root_path}/data/merged_video"
image_root_path = f"{root_path}/data/image"
upsampled_image_root_path = f"{root_path}/upsampled/image"
upsampled_video_root_path = f"{root_path}/upsampled/video"


def get_video_path(video_id, ratio, extension):
    return f"{video_root_path}/{video_id}/{ratio}/{video_id}.{extension}"


def get_upsampled_video_path(video_id, ratio, extension):
    os.makedirs(f"{upsampled_video_root_path}", exist_ok=True)
    return f"{upsampled_video_root_path}/{video_id}_{ratio}.{extension}"


def get_image_path(video_id, ratio):
    os.makedirs(f"{image_root_path}/{video_id}/{ratio}", exist_ok=True)
    return f"{image_root_path}/{video_id}/{ratio}"


def get_upsampled_image_path(video_id, ratio):
    os.makedirs(f"{upsampled_image_root_path}/{video_id}/{ratio}",
                exist_ok=True)
    return f"{upsampled_image_root_path}/{video_id}/{ratio}"


if __name__ == "__main__":
    pass