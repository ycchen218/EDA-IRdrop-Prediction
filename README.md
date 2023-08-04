# EDA IR Drop Prediction
## Update
- 2023/08/04<br>
  Add the model overview and brief description.
- 2023/6/24<br>
  Add the IR_drop_train.py file. <br>
  Delete model weight<br>
  Get model weight at [GoogleDrive](https://drive.google.com/drive/folders/1ZscqJsUbKuphnyAL7LrQXB62dC6uc9fs?usp=sharing)
## Introduce
IR drop analysis is an essential step for IC design, evaluating the power integrity of a chip power delivery network.
This is a deep-learning-based EDA tool for predicting the IR Drop heatmap, I achieve high precision and accuracy by inputting the power features. 
Additionally, for one prediction process only costs about 2 seconds.
## Requirement
1. python3.8
2. scipy
3. matplotlib
4. numpy
5. opencv
6. pandas
7. pytorch 1.12.0
## Model Overview
|feature         | discribe                   |
|------------------|-----------------------|
| $p_{i}$        | Internal power               |
| $p_{s}$ | Switching power              |
| $p_{sac}$ | Toggle rate scaled power         |
| $p_{all}$ | $p_{i}+p_{s}+p_{sac}$     |
|$p_{t}$    |The power of an instance at each time step | 

For more feature information V. A. Chhabria *et al.* **[1]** have discribe the detail of the feature extraction process.<br>
The CE-FPN model is proposed by Y. Luo *et al.* **[2]**.
![image](https://github.com/ycchen218/EDA-IRdrop-Prediction/blob/master/git_image/IRdrop_model.png)
1. The purple CNN block is the CoordConv.
2. SSF is the Sub-pixel Skip Fusion.
3. SCE block is the Sub-pixel Context Enhancement.
4. CAG block is the Channel Attention Guided Module.
## Train
```markdown
python IR_drop_train.py
```
--root_path: The path of the data file <br>
--batch_size: The batch size figure <br>
--num_epochs: The training epochs <br>
--learning_rate: learning rate [0,1] <br>
--weight_path: The path to save the model weight <br>
--fig_path: The path of the figure file <br>
## Predict
```markdown
python IR_drop_predict.py
```
--data_path: The path of the data file <br>
--fig_save_path: The path you want to save figure <br>
--weight_path: The path of the model weight <br>
--output_path: The path of the predict output with .npy file <br>
--irdrop_threshold: irdrop_threshold [0,1] <br>
--device: If you want to use gpu type "cuda" <br>
## Predict result
1. Tune your own irdrop_threshold, the defalt is 0.1 as shown in following figure.
2. The output coordinate csv file and image array npy file are in the ./output file.
3. The model predict cost time is **1.49 ~ 2.2 sec**.

![image](https://github.com/ycchen218/EDA-IRdrop-Prediction/blob/master/git_image/predict.png)
## Compare with ground truth
![image](https://github.com/ycchen218/EDA-IRDrop-Prediction/blob/master/git_image/test_compare.png)
## Cross validation while evalulate the model
ROC Curve:<br>
![image](https://github.com/ycchen218/EDA-IRDrop-Prediction/blob/master/git_image/ROC_curve.png)<br>
SSIM score: **0.863** <br>
AUC: **0.991** <br>
Percision: **0.9975** <br>
by pytorch_msssim.SSIM<br>
by the same metrics code as [CircuitNet](https://github.com/circuitnet/CircuitNet)
## Reference
```markdown
[1] V. A. Chhabria, Y. Zhang, H. Ren, B. Keller, and S. S. Sapatnekar, "Mavirec: mL-aided vectored ir-drop estimation and classification," *Design, Automation & Test in Europe Conference & Exhibition*, pp. 1825-1828, 2021.
[2] Y. Luo, X. Cao, J. Zhang, X. Cao, F. Guo, H. Shen, T. Wang, and Q. Feng, "Ce-fpn: enhancing channel information for object detection," ArXiv Computer Vision and Pattern Recognition, pp. 1-9, 2021.
```
