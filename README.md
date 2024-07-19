# Pixel-Scan

This project consists of two parts: 
1) segmentation of the pancreas on 3D CT images using machine learning
2) an algorithm for predicting pathology of the organ based on the resulting segmentation mask without using machine learning

This repository presents the code for the first part only.


## Segmentation Task

The aim of this step was to segment the pancreas and all its regions. The resulting number of segmentation classes was 4 classes of pancreatic regions + background class. PyTorch implementation of the original **3D U-Net** architecture (https://arxiv.org/pdf/1606.06650v1.pdf) was used as the underlying machine learning model.

3D segmentation task is computationally intensive due to the large size of the medical input data. Therefore, several approaches to **preprocessing the data** were explored:
1) Compressing original images of size (512, 512, N) to size (128, 128, 128) with quality degradation.

![image](https://github.com/user-attachments/assets/a207e280-024d-46cd-b195-11f5b2129786)

2) Cropping original images to the region of interest of size (256, 256, 256) without resolution loss and then compressing them to size (128, 128, 128).

![image](https://github.com/user-attachments/assets/e3179258-cb2d-4f3d-aa8a-aac2af7ddb71)

3) Partitioning the cropped images into independent patches of size (64, 64, 64).

![image](https://github.com/user-attachments/assets/6bfc73a9-bd22-4113-9934-662e2bc478f9)


## Dataset

Total three datasets were used:
- Private dataset with abdominal CT scans
- Medical Segmentation Decathlon open dataset
- MICCAI FLARE 2023 open dataset

Open datasets containing only pancreas and tumor tags were further segmented into 4 classes.


## Results

<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <table>
        <tr>
            <th rowspan="2">Algorithm</th>
            <th colspan="4">Dice Score</th>
        </tr>
        <tr>
            <th>Head</th>
            <th>Neck</th>
            <th>Body</th>
            <th>Tail</th>
        </tr>
        <tr class="highlight">
            <td>DownSampled U-Net</td>
            <td>0.78</td>
            <td>0.62</td>
            <td>0.74</td>
            <td>0.68</td>
        </tr>
        <tr>
            <td>Cropped U-Net</td>
            <td>0.66</td>
            <td>0.45</td>
            <td>0.61</td>
            <td>0.52</td>
        </tr>
        <tr>
            <td>Patch U-Net</td>
            <td>0.35</td>
            <td>0.18</td>
            <td>0.28</td>
            <td>0.25</td>
        </tr>
        <tr>
            <td>DownSampled U-Net with Class Weights</td>
            <td>0.32</td>
            <td>0.23</td>
            <td>0.34</td>
            <td>0.23</td>
        </tr>
        <tr>
            <td>Cropped U-Net with Class Weights</td>
            <td>0.16</td>
            <td>0.10</td>
            <td>0.13</td>
            <td>0.05</td>
        </tr>
    </table>
</body>
</html>
