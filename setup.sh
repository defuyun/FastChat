sudo apt install ubuntu-drivers-common alsa-utils
sudo apt install nvidia-driver-525
sudo reboot
sudo apt install nginx
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
# in base conda environment
pip install huggingface transformers einops accelerate scipy bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit
python -m fastchat.serve.controller &> controller.log&
python -m fastchat.serve.model_worker --model-path tiiuae/falcon-40b-instruct &> worker.log&
# to run 8bit (requires 40+gb of GPU mem)
# python -m fastchat.serve.model_worker --model-path tiiuae/falcon-40b-instruct --load_8bit &> worker.log&
python -m fastchat.serve.simple_server --host localhost --port 8000 &> server.log&
sudo cp -r nginx /etc/nginx
sudo systemctl restart nginx