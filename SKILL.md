---
name: photo-detail-edit
description: Edit a photo with GPT Image using the user's prompt plus preservation constraints, then learn structure-gated local RGB mappings and apply them to the native original while retaining the edited background and transparency. Use for photo style or background edits that should retain real detail and target color and lighting.
---

# Photo Detail Edit

用户提供原图和编辑 prompt。本 skill 完成整条流程：**追加保真约束 → 内置图片编辑 → 小图结构匹配 → 学习局部 RGB 变换 → 应用于原尺寸原图 → 成片与局部对照**。不要只完成生成或给出操作建议。

## 输入与边界

- 需要能读取的原始图片文件。先检查文件像素、字节数、EXIF 方向和透明度；视觉预览可能被缩小，不能作为“原始大图”。若只有会话预览且无法取得原文件，请用户提供原文件或装有原图的 ZIP。不要从已丢失的细节中声称恢复了真实信息。
- 用户只需提供图片和 prompt；由 agent 判断保留区域、制作分割提示并检查遮罩，不要求用户先手动画遮罩。
- 保留用户明确要求的改动。背景替换、去背景、换衣、删除饰物等对应区域不得被后续恢复步骤改回；对明确改变的物体或结构，建立排除区域。
- 本流程面向摄影修图。大幅改变姿势、人物身份或整体重绘可能无可靠对应，须如实保留生成结果并说明恢复有限。

## 工作流程

### 1. 建立本次任务目录

在当前工作区建立新的输出目录，保留原图，不覆盖原文件或已有成片。保存原始用户 prompt、实际发送的完整 prompt 和生成的底图。附件 ZIP 解压时核验路径，不覆盖已有文件。

若依赖未安装，在当前项目的虚拟环境中安装 `scripts/requirements.txt`；优先复用已有兼容环境，或用 `uv venv` / `uv pip install --python ... -r ...`。无需系统级安装。脚本运行时不联网、不下载模型。

### 2. GPT Image 编辑

- 先通过 `view_image` 查看原始本地文件。调用本会话可用的内置 `image_gen` 编辑工具，以**原始文件**为编辑输入，不以前一轮生成图代替原图。
- 实际 prompt = 用户原文 + [固定保真约束](references/preservation-prompt.txt)。固定段作为追加约束，不替换、稀释用户目标；全身构图要求仅适用于原本就是全身构图的照片。保存这个准确的拼接结果。
- 去背景/抠图/透明背景请求设置 `transparent_background: true`；替换为新背景则为 false。其他参数遵循当前工具能力，不虚构质量、尺寸或模型版本参数，不承诺 prompt 能保证像素或分辨率保真。
- 将生成文件复制到本次任务目录。记录真实输出尺寸与大小。默认一次生成；仅在工具失败或明显漏掉用户主要要求时修正并重试一次，避免无止境再生成。不要擅自切换到付费 API/CLI。

### 3. 为本次图片建立保留区域

查看原图和生成图。按 [区域配置与运行说明](references/regions-and-running.md) 创建本次 `regions.json`，分别指定原图和生成图的主体遮罩或 GrabCut 种子；坐标从当前图片判断，不复制上次人物坐标。脚本会将配置坐标按各自图片的 EXIF 显示方向解释。

遮罩应覆盖需要保真的共同主体，排除背景和用户有意改动的部位。透明前景可利用 alpha，但仍须排除用户有意重绘的区域。`fill_holes` 默认关闭，物体把手、姿态形成的空隙可能是真实背景，不能盲目填洞。

至少选择两个人物/物体细节位置做局部对照；照片中若有脸、印花、毛呢/织物，优先这些位置。配置来自 agent 的视觉判断，不是固定模板。

### 4. 分析、检查，再融合

使用同一组参数，先 `--stage analyze`，再 `--stage fuse`：

```text
<python> <skill>/scripts/restore_detail.py --source <original> --base <generated> --regions <regions.json> --out <job>/restored --stage analyze
```

默认 `--method local-rgb --structure-channel auto`。分析阶段将原图缩到共同尺寸，验证几何对应，降低颜色、对比度和细纹理对结构判断的影响，在可靠区域拟合受约束、保持单调的局部 RGB 颜色曲线。普通多色照片使用亮度结构；强单色光自动选择相应通道，两图使用相同表示，不固定红通道。

检查 `base-mask.png`、`source-mask.png`、`confidence-overlay.png` 和 `report.json`，确认实际方法为 `local-rgb`，结构通道和匹配范围合理。背景不得被选进恢复区域；遮罩若明显错误，修正区域配置后重新分析。不要通过放宽匹配阈值强行提高覆盖率。

```text
<python> <skill>/scripts/restore_detail.py --source <original> --base <generated> --regions <regions.json> --out <job>/restored --stage fuse
```

脚本将局部曲线参数平滑上采样，应用于配准后的原尺寸原图，让真实细节经过目标色调和光影的变换。映射不可靠的位置沿用放大的生成图；背景、透明度和明确排除的编辑区域保持一致。没有足够共享特征时不传入原图内容。主体色调与光影是拟合结果，不能承诺逐像素一致，也不能将原图尺寸称为完整恢复原图画质。

不默认叠加 Lab 灰度高频，也不默认使用自由度过高的任意矩阵或完整 3D LUT。`--method lab-detail` 仅保留用于用户明确要求的旧方案比较；不要静默退回旧方法。

### 5. 检查并交付

- 检查最终预览与至少两个原始像素尺度的局部对照，分别核对真实纹理、颜色和大范围光影，关注重影、黑块、暗部发灰、边缘和接地阴影。不要仅凭纹理相关度或匹配覆盖率宣告画质改善。必要时收紧排除区域，再运行一次分析与融合。
- 确认 `report.json` 中背景范围像素变动为 0、alpha 保留、PNG 可读取。精确背景一致性针对无损 PNG，JPEG 压缩可能改变像素。
- 返回可见预览、完整尺寸无损 PNG 和局部对照链接。无透明度时另交付 JPEG；透明图不通过 JPEG 交付。
- 简述输出尺寸、实际恢复范围和明显限制；提供实际完整 prompt 的文件链接。不要把插值放大的像素数量当成新增真实细节。

## 调用示例

“使用 `$photo-detail-edit` 处理这张原图：净化背景和光影，使它具有杂志 studio 摄影的质感。”

“使用 `$photo-detail-edit` 去掉背景，输出透明 PNG，保留人物和衣物细节。”
