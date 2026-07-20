# SmartSens SSNE AI Demo

本目录是 SmartSens A1/SC132GS 开发板上的多功能演示工程。程序入口为 `ssne_ai_demo/main.cpp`，启动后通过主菜单选择 1～8 号功能。

## 功能概览

### 1. 人脸检测

- 模型：`app_assets/models/face_640x480.m1model`。
- 摄像头裁剪区域：720×540，原图坐标为 `x=0..720, y=370..910`。
- 模型输入：640×480 灰度图；检测结果映射回 720×1280 原图。
- 按 `q/Q` 返回主菜单。

### 2. 表情识别

- 模型：`emotion_4class.m1model`。
- 输入区域：原图 `x=40..680, y=320..960` 的 640×640 Y8 灰度图。
- 模型输入：112×112、RGB 三通道、uint8、像素范围 0～255。
- 预处理：灰度图缩放到 112×112，并将同一灰度通道复制到 R/G/B 三个通道；不是补零，也不做 `/255` 归一化。
- 输出 4 类：`SURPRISE`、`HAPPY`、`SAD`、`NEUTRAL`。
- 通过图像质量检查、时序缓冲、置信度/间隔阈值和连续帧确认稳定结果；不确定结果在超时后回退为中性。
- 按 `q/Q` 返回主菜单。

### 3. 手势识别

- 模型：`gesture.m1model`。
- 输入 ROI：原图 `x=40..680, y=320..960` 的 640×640 区域。
- 模型输入：224×224，支持 6 个模型类别。
- 预处理包含背景减除；可按 `c/C` 开关 CLAHE 对比度增强。
- 按 `q/Q` 返回主菜单。

### 4. 目标检测

- 模型：`yolov8n_object_detection.m1model`。
- 输入：整幅 720×1280 灰度图，板端送入 640×480 灰度模型输入，并执行 letterbox 与坐标逆映射。
- 类别名称由 `app_assets/cls.yaml` 提供，当前为 20 类 COCO 风格类别。
- 使用 YOLOv8 多尺度输出、置信度筛选和 NMS，OSD 显示检测框、类别和分数。
- 按 `q/Q` 返回主菜单。

### 5. 光流避障

- 纯 CPU 实现，不依赖 NPU 模型。
- 处理区域：720×540，来自原图 `y=370..910`。
- 算法链路：FAST 特征点 → Lucas-Kanade 稀疏光流 → 全局运动补偿 → 区域中值统计 → TTC/散度风险判断。
- 当前参数包括 TTC 阈值 `2.0s`、散度阈值 `0.45`，不是 1.0s。
- 支持标准模式和 Arduino 模式；Arduino 模式通过 Arduino Bridge 输出避障反馈。
- 按 `p/P` 暂停/继续，按 `q/Q` 返回主菜单。

### 6. 剪刀石头布（RPS）

- 模型：`paper.m1model`，模型输出 ROCK、PAPER、SCISSORS 三类分数。
- 输入源：整幅 720×1280 Y8 灰度图。
- 预处理：缩放到 360×640，与历史帧做帧差，再进行阈值筛选、增益放大和像素数量运动门控。
- `IDLE` 不是模型类别，而是由运动门控和状态机生成。
- 状态机处理空闲、前摇、预测锁定和显示保持状态；连续帧加权投票后锁定用户手势，并生成 AI 反制手势：ROCK→PAPER、PAPER→SCISSORS、SCISSORS→ROCK。
- OSD 左侧显示用户手势，右侧显示 AI 反制手势；串口在锁定结果变化时输出结果。
- 支持 `show`、`help`、`threshold N`、`idle_pixels N`、`gap N`、`gain N`、`windup N`、`vote N`、`q` 等运行时命令。

### 7. 速度检测

- 模型：`yolov8n_speed_detection.m1model`。
- 摄像头输入区域：720×720，处理窗口位于竖屏画面的下方区域。
- 模型输入：320×320 灰度图。
- 检测类别：`CAR`、`TRUCK`、`BUS`；类别名称和玩具车辆尺寸在板端后处理中配置。
- 后处理包含置信度筛选、NMS、目标关联、运动方向判断、像素位移到厘米速度换算、平滑和静态杂物抑制。
- `CAR` 默认使用 0.45 置信度阈值，其他车辆默认使用 0.35；静态杂物需持续满足静止条件后才会被抑制。
- 按 `f/F` 后回车显示最近 10 帧 FPS 和运行平均 FPS，按 `q/Q` 退出。

### 8. 追焦功能

进入后有三个子功能：

1. **MotionGuard CPU 多目标风险追踪**：支持居家守护和路侧监控两套场景参数，使用 CPU 运动分析、多目标跟踪和风险状态机。
2. **EyeDet-S + FaceID-S 智能追焦**：使用 `eyedet_s.m1model` 检测双眼，以几何区域构造人脸 ROI，再使用 `faceid_s.m1model` 完成临时身份录入和识别。
3. **EyeDet-Flash + FaceID-S 高帧率追焦**：使用 `eyedet_flash.m1model`，输入为 320×480 灰度图；FaceID-S 流程保持不变。

追焦子功能按键：`p/P` 暂停/继续，`r/R` 重置锁定目标，`e/E` 录入 `id_tmp`，`c/C` 清除临时身份，`q/Q` 退出当前子功能。

## 模型与资源

当前模型文件为：`emotion_4class.m1model`、`eyedet_flash.m1model`、`eyedet_s.m1model`、`face_640x480.m1model`、`faceid_s.m1model`、`gesture.m1model`、`paper.m1model`、`yolov8n_object_detection.m1model`、`yolov8n_speed_detection.m1model`。

OSD 资源包括 `*.ssbmp` 位图以及 `colorLUT.sscl`、`shared_colorLUT.sscl`；`cls.yaml` 用于目标检测类别名称映射。

## 源码结构

```text
main/
├── README.md
└── ssne_ai_demo/
    ├── main.cpp
    ├── CMakeLists.txt
    ├── cmake_config/Paths.cmake
    ├── app_assets/          # 模型、OSD 位图、LUT 和 cls.yaml
    ├── common/              # 公共图像管线、OSD 和工具类
    ├── modules/             # 各业务模块及 Arduino Bridge
    └── scripts/run.sh
```

## 运行

板端进入包含 `app_assets/` 的目录后执行：

```sh
cd /app_demo
./ssne_ai_demo
```

退出当前模块后程序返回主菜单；主菜单选择 `0` 退出整个程序。

## 工程流程图

<p align="center">
  <img src="project_flowchart.svg" alt="SSNE AI Demo 项目整体流程图" width="85%"/>
</p>
