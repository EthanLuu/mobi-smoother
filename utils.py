import os
import cv2
import math
import numpy as np
import logging
from pathlib import Path
# import tensorflow as tf


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


def get_psnr(img1, img2):
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    mse = np.mean((img1 / 255. - img2 / 255.)**2)  # type: ignore # 均方差
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_video_score(vu1, vu2):
    # 抽取前 10 帧作为超分的效果
    score = 0
    cnt = 0
    for idx in range(vu1.video_info['nb_frames']):
        if idx >= 100: break
        f1 = vu1.read_frame()
        f2 = vu2.read_frame()
        score += get_psnr(f1, f2)
        cnt += 1

    return round(score / cnt, 2)


def get_logger(fpath):
    Path(fpath).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name='r')  # set root logger if not set name
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # output to file by using FileHandler
    fh = logging.FileHandler(fpath + "log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # output to screen by using StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

if __name__ == "__main__":
    # vu1 = VideoUpsampler()
    # vu1.read_video("1", "1", "1_720p.mp4")
    # vu2 = VideoUpsampler()
    # vu2.read_video("1", "2", "1_1080p.mp4")
    # print(get_video_score(vu1, vu2))
    img1 = cv2.imread("0801.png")
    img2 = cv2.imread("0801x2.png")
    img3 = cv2.imread("0801x4.png")
    # img4 = read_img("0801x4m.png")
    # img5 = cv2.imread("0801x4mx.png")
    print(get_psnr(img2, img1))
    print(get_psnr(img3, img1))
    # print(psnr(img1, img4))
    # print(get_psnr(img5, img1))
    pass


def to_obj(obj: object, **data):
    obj.__dict__.update(data)