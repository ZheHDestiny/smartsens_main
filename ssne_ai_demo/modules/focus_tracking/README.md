# 追焦模块

本目录只放追焦功能的板端业务入口和模块私有实现。追焦实时链路全部使用 C++：

- `focus_tracking_main.cpp`：暴露给顶层 `main.cpp` 的功能入口。
- `src/focus_tracker.cpp`：无 NPU 传统视觉追踪实现，以及 MobileNet NPU 模式的接口占位。

运行方式：在主菜单选择 `8. 追焦功能` 后，会进入追焦子菜单：

- `1. 传统视觉追焦`：不加载模型，直接使用 CPU 高帧率模板追踪。
- `2. MobileNet NPU追焦`：预留模型选目标入口，后续加载 `focus_mobilenet.m1model` 后低频调用 NPU，输出 ROI 后由传统追踪器高频接管。
- `0. 返回主菜单`。

子功能运行中按 `Q/q` 只退出当前追焦子功能并返回追焦子菜单；回到追焦子菜单后再输入 `0`，才返回顶层主菜单。

实时运行时不会调用 shell 或 Python 脚本。后续若需要补充脚本，只能作为开发辅助：

- shell：用于板端批量启动、采集样本或整理日志，不参与每帧追焦处理。
- Python：用于 PC 端离线评估、画轨迹图、检查训练数据，不在板端实时路径中运行。

模型文件统一放在 `../../app_assets/models/`，追焦 MobileNet 模型建议命名为 `focus_mobilenet.m1model`。
