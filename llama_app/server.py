import subprocess


def run_server():
    print("Model is running...")
    args = ("../Resources/server", "-m", "../Resources/llama-2-13b-chat.ggmlv3.q4_0.bin", "-c", "2048")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    print("Model is done")
