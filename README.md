## MacOS (AppleSilicon)安装tensorflow

### miniforge3安装

1. download install script
```shell
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
sh Miniforge3-MacOSX-arm64.sh

conda --version
```

2. build conda env

```shell
conda create --name python38 python=3.8
conda activate python38
```

3. download tensorflow for AppleSilicon

```shell
wget https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_macos-0.1alpha2.tar.gz
tar zxf tensorflow_macos-0.1alpha2.tar.gz
```

4. install tensorflow

```shell
libs="/Users/kk/Downloads/tensorflow_macos/arm64/"
env="/Users/kk/miniforge3/envs/python38/"
conda install cached-property
conda install six

# 安装相关依赖
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/grpcio-1.33.2-cp38-cp38-macosx_11_0_arm64.whl"
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/h5py-2.10.0-cp38-cp38-macosx_11_0_arm64.whl"
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/tensorflow_addons_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl"
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/numpy-1.18.5-cp38-cp38-macosx_11_0_arm64.whl"
# 安装相关库
conda install -c conda-forge -y absl-py
conda install -c conda-forge -y astunparse
conda install -c conda-forge -y gast
conda install -c conda-forge -y opt_einsum
conda install -c conda-forge -y termcolor
conda install -c conda-forge -y typing_extensions
conda install -c conda-forge -y wheel
conda install -c conda-forge -y typeguard
pip install wrapt flatbuffers tensorflow_estimator google_pasta keras_preprocessing protobuf
# 安装macOS tensorflow库
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/tensorflow_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl"
#安装 tensor board
pip install tensorboard

```


5. check tensorflow

```python
import tensorflow as tf
print(tf.__version__)
```