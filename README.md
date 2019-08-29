# 表情识别

使用卷积神经网络训练表情识别算法，实现实时读取识别摄像头人脸并识别表情

## 实验环境

```
tensorflow-gpu=1.12
CUDA=9.2
opencv=3.4
Linux + GPU
```

## 实验

### 训练

```bash
python train_test.py \
--mode train \
--dataset_path data/FER \
--output_path data/model \
--max_step 2000
```
训练结束将在model文件夹下生成cnn_emotion.ckpt文件

### 测试

```bash
python train_test.py \
--mode test \
--test_model_folder testmodel
```
读取上一步生成的模型参数并测试模型精度

### 实时识别人脸表情

<<<<<<< HEAD
=======
```bash
python show_face.py
```

![](https://github.com/wangz49777/Emotion_recognition/blob/branch1/src/example.png)

按q键关闭窗口

### 数据集

#### FER 和 FERPlus

![FERvsFER+](https://github.com/wangz49777/Emotion_recognition/blob/branch1/src/FER%2BvsFER.png)

FER(上)和FERPlus(下)

## 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE](LICENSE)

## 鸣谢

