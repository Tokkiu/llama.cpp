import subprocess

source = "../Resources/"
# source = "./"
model = "llama-2-13b-chat.ggmlv3.q4_0.bin"
def run_server():
    print("Model is running...")
    args = (source + "server", "-m", source + model, "-c", "2048", "--port", "31080")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    print("Model is done")
