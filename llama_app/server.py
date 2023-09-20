import subprocess
import os, time, requests
from threading import Thread
# from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
# from tqdm. import tqdm

source = "../Resources/"
# source = "./"
model = "llama-2-13b-chat.ggmlv3.q4_0.bin"
# model = "llama_13B"
# download_url = "https://cdn-lfs.huggingface.co/repos/cd/43/cd4356b11767f5136b31b27dbb8863d6dd69a4010e034ef75be9c2c12fcd10f7/f79142715bc9539a2edbb4b253548db8b34fac22736593eeaa28555874476e30?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27llama-2-13b-chat.ggmlv3.q4_0.bin%3B+filename%3D%22llama-2-13b-chat.ggmlv3.q4_0.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1695104951&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5NTEwNDk1MX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9jZC80My9jZDQzNTZiMTE3NjdmNTEzNmIzMWIyN2RiYjg4NjNkNmRkNjlhNDAxMGUwMzRlZjc1YmU5YzJjMTJmY2QxMGY3L2Y3OTE0MjcxNWJjOTUzOWEyZWRiYjRiMjUzNTQ4ZGI4YjM0ZmFjMjI3MzY1OTNlZWFhMjg1NTU4NzQ0NzZlMzA%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Y9KAP-DU5FOEIlD-T%7EHYHp49cno659NOpkvWG%7EMiMNjsWyMw6IG0dnykJEUlWqVnIL6uiRcteIRfvToxt5BHPDIg7MuRNR6YZcKiHuzt1UelIQr3ESEuVUtoEC6CoHiH6mTKxbPgOmldml5PnSy3hxKTh3I2DMXozXqmjanETKJnjIcbnMrT4VLgOwN2k0IBFsV29OxswuyLM8AM1SyeBmVN4fgbFNijqul4sQNUVW-nZ7oRrRNgbn3TYRy9WQ82AAXiqmv1fmvfe5TlESj1h5j413gsUnTl-4cJuiVIJ6vLVxzrhg5Tmz-1-oA9X4cT4i47jCPMwd4PUZKdTOO9KA__&Key-Pair-Id=KVTP0A1DKRTAX"
download_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin "


def run_server(path=""):
    # if len(path) == 0:
    path = source + model
    print("Model is running...", path)
    args = (source + "server", "-m", path, "-c", "2048", "--port", "31080")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    print("Model is done")



class ModelStatus:
    def __init__(self, name):
        self.is_downloaded = False
        self.is_running = False
        self.name = name
        self.location = source + name
        self.download_process = 0.0


# 进度条模块
class Downloader:
    def __init__(self):
        self.content_size = 0
        self.progress = 0.0
        self.start = time.time()
        self.is_downloading = False
        self.path = source
        self.url = download_url
        self.chunk_size = 1024
        self.size = 0

    def downloaded(self):
        return not self.is_downloading and self.progress == 100.

    @property
    def download_lib(self):
        downloaded_model_path = snapshot_download(
            repo_id="TheBloke/Llama-2-13B-chat-GGML",
            library_name="llama-2-13b-chat.ggmlv3.q4_0.bin",
            ignore_patterns=["*q2_*", "*q3_*","*q4_1*", "*q4_K*","*q5_*", "*q6_*", "*q8_*", ],
            tqdm_class=TqdmUpTo
            # use_auth_token=True
        )
        print(downloaded_model_path)
        return downloaded_model_path

    def download_impl(self, response, filepath, status: ModelStatus):
        with open(filepath, 'wb') as file:
            for data in tqdm(response.iter_content(chunk_size=self.chunk_size)):
                file.write(data)
                self.size += len(data)
                # print('\r' + '[Process]:%s%.2f%%' % (
                #     ' ' * int(self.size * 50 / self.content_size), float(self.size / self.content_size * 100)), end=' ')
                self.progress = float(self.size / self.content_size * 100)
                status.download_process = self.progress

        end = time.time()
        print('Download completed!,times: %.2f秒' % (end - self.start))
        self.is_downloading = False
        status.is_downloaded = True
        model_t = Thread(target=run_server, args=())
        model_t.start()
        status.is_running = True

    def download(self, status):
        if self.is_downloading or self.progress == 100.:
            return True

        self.is_downloading = True
        if not os.path.exists(self.path):  # 看是否有该文件夹，没有则创建文件夹
            os.mkdir(self.path)
        response = requests.get(self.url, stream=True, allow_redirects=True, )  # stream=True必须写上
        self.content_size = int(response.headers['content-length'])  # 下载文件总大小
        try:
            if response.status_code == 200:  # 判断是否响应成功
                print('Start download,[File size]:{size:.2f} MB'.format(
                    size=self.content_size / self.chunk_size / 1024))
                model_path = self.path + model
                t = Thread(target=self.download_impl, args=(response, model_path, status))
                t.start()
            print("Download fail", response.status_code)
        except Exception as e:
            print(e)
            print("Download fail")
            self.is_downloading = False
            return False

        return True
