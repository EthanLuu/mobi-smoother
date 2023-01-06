import json
import os
import pandas as pd
import requests

root_path = os.getcwd()
data_path = root_path + "/data/"
video_path = data_path + "video/"
json_file_name = "aweme_list.json"
video_file_name = "video2.csv"

proxies = {'http': "http://127.0.0.1:7890",
           'https': "http://127.0.0.1:7890"}


def read_list():
    video_list = []
    json_path = data_path + json_file_name
    with open(json_path, "r") as f:
        data = json.load(f)
        item_list = data['itemList']
        for item in item_list:
            video_info = item['video']
            resolution_info = video_info['definition']
            video_url = video_info['downloadAddr']
            video_id = video_info['id']
            video_list.append(
                {"id": video_id, "url": video_url, "resolution": resolution_info})
    return video_list


def read_list_2():
    video_list = []
    json_path = data_path + json_file_name
    with open(json_path, "r") as f:
        data = json.load(f)
        item_list = data['itemList']
        for item in item_list:
            video_info = item['video']
            print(video_info)
            if not video_info:
                continue
            resolution_info = video_info['ratio']
            video_url = video_info['play_addr']['url_list'][0]
            video_id = item['group_id']
            video_list.append(
                {"id": video_id, "url": video_url, "resolution": resolution_info})
    return video_list


def save_list(data, file_name):
    data.to_csv(data_path + file_name, index=False, sep=',')


def download_video(url, path, file_name):
    res = requests.get(url, proxies=proxies)
    print(res)
    with open(path + file_name, "wb") as video_file:
        video_file.write(res.content)


def download_list(video_list):
    for video_info in video_list:
        cur_path = video_path + \
            video_info['id'] + "/"
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
        cur_path += video_info['resolution'] + "/"
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
        url = video_info['url']
        download_video(url, cur_path, video_info['id'] + ".mp4")


def main():
    video_list = read_list_2()
    df = pd.DataFrame(video_list)
    save_list(df, video_file_name)
    download_list(video_list)


if __name__ == "__main__":
    main()
