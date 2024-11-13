# Cloud Segmentation

This repository contains the UNet model that I have trained for cloud segmentation from remote satellite images. The model is saved in the `output` directory named `unet_model.pth`. The directory also contains the evaluation metrics in the file `metrics.json`, the accuracy and loss plots, and a sample prediction plot.

## Usage
To predict the cloud segmentation for the sample image in the `output` directory, run the below command:
```bash
python prediction.py --model ./output/unet_model.pth --image ./data/img/wind1_8_1.jpg --target ./data/label/wind1_8_1.png
```

# Data
```
mingyuan he, jie zhang, xinjie zuo, et al. (2024) . Data for “Annotated dataset for training cloud segmentation neural net-works using high-resolution satellite remote sensing imagery”. V2. Science Data Bank. https://doi.org/10.57760/sciencedb.07830.
```
