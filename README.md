# Image-Calibrate

<details open>
<summary>安装</summary>

```bash
pip install opencv-python==4.1.2.30
pip install opencv-contrib-python==4.1.2.30
```

```bash
git clone https://github.com/fy-shen/Image-Calibrate.git
```

</details>

<details>
<summary>手动标点矫正图像</summary>


使用默认数据进行测试

```bash
cd Image-Calibrate
python3 calibrate_select_points.py --random-select 30
```

(1) 可以看到两幅图像: camera、pattern，图像上已经使用了提前手动标定的点位，此时在图像上可以点击鼠标左键添加点、鼠标右键删除上一个添加的点

(2) 选点完成后，键盘按 's' 进入矫正，此时会多出三幅图像: dst、dst_full、dst_hom

dst: <code>cv2.getOptimalNewCameraMatrix()</code> 中 <code>alpha=1</code> 时图像未裁剪的结果，可用于查看矫正效果是否符合预期，通常图像的黑色区域呈枕形效果较好

dst_full: <code>cv2.getOptimalNewCameraMatrix()</code> 中 <code>alpha=0</code> 时的结果，此时图像为原图分辨率，但经过了矫正与裁剪，通常是矫正完最终所使用的的图像效果

dst_hom: 将 dst_full 通过单应性变换映射到 pattern 上的叠加显示，用于查看单应性变换的效果

(3) 此时可按除 'r' 以外的键保存当前结果，或者按 'r' 进入随机取点矫正，可以直接看到随机取点后的结果

有时矫正的效果并不理想，而通过变换取点的数量和顺序会得到不同的结果，随机取点会在手动标定的所有点中随机选取 <code>--random-select</code> 个，得到想要的效果后再保存结果即可

(4) 保存的结果会自动存储在项目路径下的 '../runs/runx/' 中，假设相机图像文件名为 'xxx.jpg' 会得到以下结果

| file                             | description  |
| :------------------------------- | :----------- |
| xxx_calibrate_parm.npz           | 矫正参数，包含内参、畸变系数、外参 |
| xxx_camera_select_kpts.jpg       | 相机取点的可视化图像 |
| xxx_camera_select_kpts.npz       | 相机取点的坐标 |
| xxx_camera_select_kpts_raw.npz   | 使用随机取点时会生成，为最初手动标点的所有点坐标 |
| xxx_hom.jpg                      | dst_hom |
| xxx_hom.npz                      | 单应性矩阵参数 |
| xxx_pattern_select_kpts.jpg      | 模板图取点的可视化图像 |
| xxx_pattern_select_kpts.npz      | 模板图取点的坐标 |
| xxx_pattern_select_kpts_raw.npz  | 使用随机取点时会生成，为最初手动标点的所有点坐标 |
| xxx_result.jpg                   | dst |
| xxx_result_full.jpg              | dst_full |

(5) 输入参数解析

| parameter                   | default | explanation     |
| :-------------------------- | :------ | :-------------- |
| --camera-img-path           |         | 相机图像路径      |
| --pattern-img-path          |         | 模板图像路径      |
| --camera-kpts-path          |         | 相机图像预选关键点 |
| --pattern-kpts-path         |         | 模板图像预选关键点 |
| --camera-select-kpts-path   |         | 相机图像已选关键点 |
| --pattern-select-kpts-path  |         | 模板图像已选关键点 |
| --fisheye                   | False   | 是否为鱼眼相机    |
| --random-select             | -1      | 随机取点数量      |

预选关键点: 模板图像通常是自己画的，尺寸对应世界坐标，一些可能想要标的点是已知的，在手动取点时会在预选点附近会自动精确选择。存储 npz 时 key 值必须设定为 kpts。

已选关键点: 程序运行后会自动选择里面存储的点位，避免反复手动取点。

</details>
