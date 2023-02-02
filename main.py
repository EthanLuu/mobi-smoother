from upsampler import VideoUpsampler
import utils

if __name__ == '__main__':
    video_id = "7150995928504077569"
    ratio = "720p"
    extension = "mp4"
    model_name = "realesr-general-wdn-x4v3"
    vu = VideoUpsampler()
    vu.init_sr_model(model_name, tile=2, gpu_id="0", netscale=4)

    # 1. 提取视频所有帧
    video_path = utils.get_video_path(video_id, ratio, extension)
    vu.read_video(video_id, ratio, video_path)
    image_path = utils.get_image_path(video_id, ratio)
    # vu.extract_frames(video_path, image_path)

    # 2. 超分所有帧
    upsampled_ratio = f"{ratio}X4"
    upsampled_image_path = utils.get_upsampled_image_path(
        video_id, upsampled_ratio)

    # 3. 合并成视频
    upsmapled_video_path = utils.get_upsampled_video_path(
        video_id, upsampled_ratio, extension)
    vu.upsample_video(upsmapled_video_path)
    # vu.merge_video(upsampled_image_path, upsmapled_video_path)
