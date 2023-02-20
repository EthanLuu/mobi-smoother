import glob
import os
import time
import ffmpeg
import cv2
import numpy as np
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from models.sr_models import RealESRGANer, SRVGGNetCompact


class VideoUpsampler:

    def __init__(self):
        return

    def init_sr_model(self,
                      model_name,
                      denoise_strength=0.5,
                      tile=0,
                      tile_pad=10,
                      pre_pad=0,
                      netscale=4,
                      gpu_id="0"):
        model = None
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3,
                            num_out_ch=3,
                            num_feat=64,
                            num_block=23,
                            num_grow_ch=32,
                            scale=4)
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3,
                            num_out_ch=3,
                            num_feat=64,
                            num_block=23,
                            num_grow_ch=32,
                            scale=4)
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3,
                            num_out_ch=3,
                            num_feat=64,
                            num_block=23,
                            num_grow_ch=32,
                            scale=2)
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3,
                                    num_out_ch=3,
                                    num_feat=64,
                                    num_conv=32,
                                    upscale=4,
                                    act_type='prelu')
        elif model_name == 'realesr-general-wdn-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3,
                                    num_out_ch=3,
                                    num_feat=64,
                                    num_conv=32,
                                    upscale=4,
                                    act_type='prelu')
        if model is None:
            return

        model_path = os.path.join('weights', model_name + '.pth')
        dni_weight = None
        if model_name == 'realesr-general-wdn-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3',
                                                'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        self.upsampler = RealESRGANer(scale=netscale,
                                      model_path=model_path,
                                      dni_weight=dni_weight,
                                      model=model,
                                      tile=tile,
                                      tile_pad=tile_pad,
                                      pre_pad=pre_pad,
                                      half=True,
                                      gpu_id=gpu_id)

    def read_video(self, video_id, ratio, video_path):
        self.stream_reader = (ffmpeg.input(video_path).output(
            'pipe:', format='rawvideo', pix_fmt='bgr24',
            loglevel='error').run_async(pipe_stdin=True,
                                        pipe_stdout=True,
                                        cmd="ffmpeg"))
        probe = ffmpeg.probe(video_path)
        video_stream = [
            stream for stream in probe['streams']
            if stream['codec_type'] == 'video'
        ]
        video_info = {}
        video_info['video_id'] = video_id
        video_info['ratio'] = ratio
        video_info['height'] = video_stream[0]['height']
        video_info['width'] = video_stream[0]['width']
        video_info['fps'] = eval(video_stream[0]['avg_frame_rate'])
        video_info['nb_frames'] = int(video_stream[0]['nb_frames'])
        self.video_info = video_info
        self.video_dic = video_stream[0]

    def extract_frames(self, video_path, image_path):
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        success = True
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        print(f"Frame Extraction starts")
        start_time = time.time()
        while success:
            count += 1
            cv2.imwrite(f"{image_path}/{str(count).zfill(8)}.jpg", image)
            success, image = cap.read()
        end_time = time.time()
        print(
            f"Frame Extraction success, frames count: {count}, time consumed: {end_time-start_time}s"
        )
        return

    def merge_video(self, image_path, video_path):
        print(f"Video Merge starts")
        start_time = time.time()
        files = os.listdir(image_path)
        files.sort(key=lambda x: int(x.replace(".jpg", "")))
        shape = [0, 0]
        imgs = []
        for file_name in files:
            img_path = f"{image_path}/{file_name}"
            img = cv2.imread(img_path)
            imgs.append(img)
            shape = img.shape
        size = (shape[1], shape[0])
        os.makedirs(video_path, exist_ok=True)
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                30, size)
        for img in imgs:
            video.write(img)
        end_time = time.time()
        print(f"Video Merge success, time consumed: {end_time-start_time}s")

    def read_frame(self):
        height = self.video_info['height']
        width = self.video_info['width']
        img_bytes = self.stream_reader.stdout.read(height * width *
                                                   3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([height, width, 3])
        return img

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def upsample_images(self, output_path, outscale=4):
        pbar = tqdm(total=self.video_info['nb_frames'],
                    unit='frame',
                    desc='inference')

        for id in range(self.video_info['nb_frames']):
            raw_frame = self.read_frame()
            try:
                result, _ = self.upsampler.enhance(raw_frame, outscale)
            except RuntimeError as error:
                print('Error', error)
                print(
                    'If you encounter CUDA out of memory, try to set --tile with a smaller number.'
                )
            else:
                cv2.imwrite(f"{output_path}/{str(id + 1).zfill(8)}.jpg",
                            result)
                pbar.update(1)

    def upsample_video(self, output="upsampled.mp4", outscale=4):
        pbar = tqdm(total=self.video_info['nb_frames'],
                    unit='frame',
                    desc='inference')
        output_height = self.video_info['height'] * outscale
        output_width = self.video_info['width'] * outscale
        self.stream_writer = (ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='bgr24',
            s=f'{output_width}x{output_height}',
            framerate=self.video_info['fps']).output(
                output, pix_fmt='yuv420p', vcodec='libx264',
                loglevel='error').overwrite_output().run_async(
                    pipe_stdin=True, pipe_stdout=True, cmd="ffmpeg"))

        for _ in range(self.video_info['nb_frames']):
            raw_frame = self.read_frame()
            try:
                result, _ = self.upsampler.enhance(raw_frame, outscale)
            except RuntimeError as error:
                print('Error', error)
                print(
                    'If you encounter CUDA out of memory, try to set --tile with a smaller number.'
                )
            else:
                self.write_frame(result)
                pbar.update(1)
        return output