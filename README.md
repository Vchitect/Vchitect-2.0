# Vchitect-XL: Parallel Transformer for Text-to-Video Diffusion Models

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://vchitect.intern-ai.org.cn/' target='_blank'>Vchitect-2.0 Team<sup>1,2</sup></a>&emsp;
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>Shanghai Artificial Intelligence Laboratory&emsp;
</div>
 
 -----------------

![](https://img.shields.io/badge/VchitectXL-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-VchitectXL_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)

**:fire:The technical report is coming soon!**

## Installation

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

  ```bash
  conda create -n VchitectXL -y
  conda activate VchitectXL
  conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
  ```

### 2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

### 3. Install ``flash-attn``

  ```bash
  pip install flash-attn --no-build-isolation
  ```

### 4. Install [nvidia apex](https://github.com/nvidia/apex)

```bash
pip install ninja
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Inference

~~~bash
#easy infer
test_file=$1
save_dir=$2
ckpt_path=$3

python inference.py --test_file "${test_file}" --save_dir "${save_dir}" --ckpt_path "${ckpt_path}"

~~~