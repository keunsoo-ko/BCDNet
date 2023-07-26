![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# BCDNet


Keunsoo Ko, Yeong Jun Koh, and Chang-Su Kim

Official PyTorch Code for 
"Blind and Compact Denoising Network Based on Noise Order Learning, IEEE Trans. Image Process., 2022"

### Installation
Download repository:
```
    $ git clone https://github.com/keunsoo-ko/BCDNet.git
```
Download [pre-trained model](https://drive.google.com/file/d/1_NxPjfxS6sJ26yLrsRYUULHQIIhwIx2Q/view?usp=sharing) parameters

### Usage
Run Test for real noise dataset on the SSID dataset:
```
    $ python demo.py --data_path "Path" --model_path BCDNet_Real.pth(put downloaded model path)
```
The path of dataset should be "Path", in which consisting of "Clean" and "Noise" folders as below
```bash
├─ Path
    ├── Clean
    └── Noise
```  
