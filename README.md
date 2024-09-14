# Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://vchitect.intern-ai.org.cn/' target='_blank'>Vchitect Team</a>&emsp;
</div>
<div>
<!-- <div align="center">
    <sup>1</sup>Shanghai Artificial Intelligence Laboratory&emsp;
</div> -->
 
 
</p>
<p align="center">
    👋 Join our <a href="https://github.com/Vchitect/Vchitect-2.0/tree/master/assets/channel/lark.jpeg" target="_blank">Lark</a> and <a href="https://discord.gg/aJAbn9sN" target="_blank">Discord</a> 
</p>

---

![](https://img.shields.io/badge/Vchitect2.0-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B)




**:fire:The technical report is coming soon!**

## 🔥 Update and News
- [2024.09.14] 🔥 Inference code and [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B) are released.

## :astonished: Gallery

<table class="center">

<tr>

  <td><img src="assets/samples/sample_0_seed3.gif"> </td>
  <td><img src="assets/samples/sample_1_seed3.gif"> </td>
  <td><img src="assets/samples/sample_3_seed2.gif"> </td> 
</tr>


        
<tr>
  <td><img src="assets/samples/sample_4_seed1.gif"> </td>
  <td><img src="assets/samples/sample_4_seed4.gif"> </td>
  <td><img src="assets/samples/sample_5_seed4.gif"> </td>     
</tr>

<tr>
  <td><img src="assets/samples/sample_6_seed4.gif"> </td>
  <td><img src="assets/samples/sample_8_seed0.gif"> </td>
  <td><img src="assets/samples/sample_8_seed2.gif"> </td>      
</tr>

<tr>
  <td><img src="assets/samples/sample_12_seed1.gif"> </td>
  <td><img src="assets/samples/sample_13_seed3.gif"> </td>
  <td><img src="assets/samples/sample_14.gif"> </td>    
</tr>

</table>

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

## Inference
**First download the [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B).**
~~~bash
test_file=$1
save_dir=$2
ckpt_path=$3

python inference.py --test_file "${test_file}" --save_dir "${save_dir}" --ckpt_path "${ckpt_path}"

~~~

## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.
