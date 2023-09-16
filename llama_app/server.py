import subprocess
import os, time, requests
from threading import Thread
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
# from tqdm. import tqdm

source = "../Resources/"
# source = "./"
model = "llama-2-13b-chat.ggmlv3.q4_0.bin"
download_url = "https://cdn-lfs.huggingface.co/repos/cd/43/cd4356b11767f5136b31b27dbb8863d6dd69a4010e034ef75be9c2c12fcd10f7/f79142715bc9539a2edbb4b253548db8b34fac22736593eeaa28555874476e30?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27llama-2-13b-chat.ggmlv3.q4_0.bin%3B+filename%3D%22llama-2-13b-chat.ggmlv3.q4_0.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1695104951&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5NTEwNDk1MX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9jZC80My9jZDQzNTZiMTE3NjdmNTEzNmIzMWIyN2RiYjg4NjNkNmRkNjlhNDAxMGUwMzRlZjc1YmU5YzJjMTJmY2QxMGY3L2Y3OTE0MjcxNWJjOTUzOWEyZWRiYjRiMjUzNTQ4ZGI4YjM0ZmFjMjI3MzY1OTNlZWFhMjg1NTU4NzQ0NzZlMzA%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Y9KAP-DU5FOEIlD-T%7EHYHp49cno659NOpkvWG%7EMiMNjsWyMw6IG0dnykJEUlWqVnIL6uiRcteIRfvToxt5BHPDIg7MuRNR6YZcKiHuzt1UelIQr3ESEuVUtoEC6CoHiH6mTKxbPgOmldml5PnSy3hxKTh3I2DMXozXqmjanETKJnjIcbnMrT4VLgOwN2k0IBFsV29OxswuyLM8AM1SyeBmVN4fgbFNijqul4sQNUVW-nZ7oRrRNgbn3TYRy9WQ82AAXiqmv1fmvfe5TlESj1h5j413gsUnTl-4cJuiVIJ6vLVxzrhg5Tmz-1-oA9X4cT4i47jCPMwd4PUZKdTOO9KA__&Key-Pair-Id=KVTP0A1DKRTAX"


def run_server():
    print("Model is running...")
    args = (source + "server", "-m", source + model, "-c", "2048", "--port", "31080")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    print("Model is done")



class DownloadStatus:
    def __init__(self):
        self.b = 0
        self.bsize = 0
        self.tsize = 0
status = DownloadStatus()

class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.

    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def display(self, msg=None, pos=None):
        print("display", msg, pos)
        super().display(msg, pos)

    def moveto(self, n):
        print("move", n)
        super().moveto(n)

    def update(self, n=1):
        print("update", n)
        super().update(n)

    def __iter__(self):
        print("iter")
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                print("iter i", n)
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    print("before time")
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        print("before update")
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        status.tsize = tsize
        status.b = b
        status.bsize = bsize
        print("my\n", 100*b*bsize/tsize)

        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


# 进度条模块
class Downloader:
    def __init__(self):
        self.content_size = 0
        self.progress = 0.0
        self.start = time.time()
        self.status = status
        self.is_downloading = False
        self.path = "./models/"
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

    def download_impl(self, response, filepath):
        with open(filepath, 'wb') as file:
            for data in tqdm(response.iter_content(chunk_size=self.chunk_size)):
                file.write(data)
                self.size += len(data)
                # print('\r' + '[Process]:%s%.2f%%' % (
                #     ' ' * int(self.size * 50 / self.content_size), float(self.size / self.content_size * 100)), end=' ')
                self.progress = float(self.size / self.content_size * 100)
                # print('progress', self.progress)

        end = time.time()
        print('Download completed!,times: %.2f秒' % (end - self.start))
        self.is_downloading = False

    def download(self):
        if self.is_downloading or self.progress == 100.:
            return True

        self.is_downloading = True
        if not os.path.exists(self.path):  # 看是否有该文件夹，没有则创建文件夹
            os.mkdir(self.path)
        response = requests.get(self.url, stream=True, allow_redirects=True)  # stream=True必须写上
        self.content_size = int(response.headers['content-length'])  # 下载文件总大小
        try:
            if response.status_code == 200:  # 判断是否响应成功
                print('Start download,[File size]:{size:.2f} MB'.format(
                    size=self.content_size / self.chunk_size / 1024))
                model_path = self.path + model
                t = Thread(target=self.download_impl, args=(response, model_path))
                t.start()
            print("Download fail", response.status_code)
        except Exception as e:
            print(e)
            print("Download fail")
            self.is_downloading = False
            return False

        return True
