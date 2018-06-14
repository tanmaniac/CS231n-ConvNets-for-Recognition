# CS231n - Convolutional Neural Networks for Visual Recognition

Course syllabus, Spring 2017, with lecture videos, notes, and assignments: http://cs231n.stanford.edu/2017/syllabus

## Running Solutions

The development environments, including Tensorflow and PyTorch, are included in the Dockerfiles in the `Docker` directory.

- Install Docker-CE as described for your distribution on the [Docker docs](https://docs.docker.com/install/).
    - Follow the [Optional Linux post-installation](https://docs.docker.com/install/linux/linux-postinstall/) steps to run Docker without `sudo`.

### Running with GPU support

This is the recommended way to build the Docker container, provided you have an Nvidia GPU with drivers installed. CUDA is contained within the Docker container, so it is not required to be installed on the host machine.

The container requires the `nvidia` Docker runtime to run - install nvidia-docker2 as described in the [`nvidia-docker` docs](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

```bash
cd Docker
bash build-gpu.sh
bash run-gpu.sh
```

### Building without GPU support (CPU only)

```bash
cd Docker
bash build-cpu.sh
bash run-cpu.sh
```

## Launching Jupyter notebooks

If running in Docker, the Jupyter notebooks must be run with a `0.0.0.0` IP address so they can be accessed from a host browser.

After launching the Docker container as described above: 

```bash
cd ~/assignment1    # Or assignment2 or 3
virtualenv -p python3 --system-site-packages .env   # system-site-packages option is necessary to find TF/PyTorch
source .env/bin/activate
pip3 install -r requirements.txt
jupyter notebook --ip 0.0.0.0 --no-browser
```