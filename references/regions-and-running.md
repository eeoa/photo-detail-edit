# 区域配置与脚本运行

`scripts/restore_detail.py` 需要 Python 3.11+、NumPy、Pillow 和 OpenCV；依赖见同目录的 `requirements.txt`。脚本不调用任何生成模型。路径通过参数传入，不依赖安装机器、当前项目或当前照片的文件名。

## regions.json

配置包含 `source`（原图）、`base`（生成图）和 `comparisons`。所有坐标都在对应图片经过 EXIF 旋转后的显示方向中，归一化为 `[0,1]`：左上角 `[0,0]`，右下角 `[1,1]`。

每个输入可选以下方式之一：

- `mask`：前景遮罩文件路径，相对路径以 regions.json 所在目录为准。灰度 >=128 为允许恢复的主体，其他位置保留生成图。遮罩只做最近邻缩放。
- `box` 加 `fg_polygons`：`box` 为 `[left,top,right,bottom]`，它之外为确定背景；`fg_polygons` 为主体内部的确定前景，多块多边形覆盖不同材质。GrabCut 扩展到主体边缘。可用 `bg_polygons` 指定空隙或其他确定背景。
- 若图已有透明度，可省略种子和 mask，使用 alpha >=250 的区域作为前景。仍可添加 `exclude_polygons`。

两种输入都支持 `exclude_polygons`：禁止恢复的多边形列表，优先级高于前景种子、遮罩和填洞。用户要求换掉的服装、删除的物品、新画出的内容应包含在生成图的排除区域中。

`fill_holes: true` 仅在检查确认封闭孔洞都是主体（例如浅色丝巾被误判为背景）时使用；默认 false。`bg_polygons` 和 `exclude_polygons` 即使填洞也保持排除。透明边缘自动排除，不把原背景混进半透明发丝。

以下只说明字段结构；**运行时必须按实际图片替换几何信息**：

```json
{
  "source": {"mask": "source-subject-mask.png"},
  "base": {"mask": "base-subject-mask.png", "exclude_polygons": []},
  "comparisons": [
    {"name": "fabric", "center": [0.45, 0.55]},
    {"name": "print", "center": [0.55, 0.4]}
  ]
}
```

GrabCut 配置字段结构为 `{"box": [l,t,r,b], "fg_polygons": [[[x,y], ...]], "bg_polygons": [], "exclude_polygons": [], "fill_holes": false}`。`fg_polygons` 必须位于实际主体内部；确定背景与确定前景至少各有一些像素。需要非矩形排除时使用多边形，不用一个覆盖大半主体的粗框。

## 运行与产物

先以 `--stage analyze` 运行并检查遮罩/权重图。正确后同参数运行 `--stage fuse`。原图、底图或区域配置一旦改变必须重新分析；脚本校验 SHA-256，拒绝复用失效的配准。

通用参数：`--source`、`--base`、`--regions`、`--out` 都是必填路径。`--work-long-edge` 默认 1800，限制配准工作图大小；最终画布仍是原图尺寸。支持 `--stage all`，适合已经审查过的配置或自动化回归。

默认 `--method local-rgb --structure-channel auto`。`--method lab-detail` 仅供明确要求的旧方案对照，不自动作为备用方法。更新前的分析缓存需要重新运行分析，不能直接用新默认方法回放。

`auto` 在受保护前景中检查目标图的线性 RGB 通道能量：若最强通道超过次强通道 4 倍，用两图的同一个该颜色通道进行结构比较，否则使用两图的线性亮度。可显式选择 `L`、`R`、`G` 或 `B`，但改变后必须重新分析。它是结构表示选择规则，不是照明光谱的精确估计。

主要输出：

- `detail-restored.png`：原图尺寸、sRGB、保留底图 alpha 的无损主文件。
- `detail-restored.jpg`：仅非透明图输出，quality 96、无色度子采样。
- `preview.png`：预览；`*-comparison.png`：局部对照，左为放大的生成底图，中为配准后的原图，右为融合结果。
- `base-mask.png`、`source-mask.png`、`confidence-overlay.png`、`detail-weight.png`：检查分割与恢复范围。
- `report.json`：输入哈希、原图/生成图/输出尺寸、实际匹配覆盖率和背景/透明度校验。
- `local-transform-field.npz`：局部 RGB 曲线参数、结构分数和恢复权重。阶段回放校验其哈希，修改后必须重新分析。

## 保真与失败判断

- 默认先将原图缩到工作图大小，在已有几何对应上比较多尺度局部自相似与无符号梯度方向，减轻曝光、对比度和细纹理的影响。该描述子不具有对任意风格变化的严格不变性，也不能代替对主体的语义判断。
- 在可靠、重叠的小区域里拟合正值、单调的逐通道 RGB 曲线。局部参数经平滑和上采样，作用到配准后的原尺寸原图；弱匹配区域与放大生成底图平滑衔接。它是受约束的局部 LUT 近似，不是任意 3D LUT。
- 颜色拟合在线性 RGB 中进行，再转换回 sRGB；不直接添加 Lab 灰度残差，不额外锐化。背景、透明度和明确排除区域沿用生成底图，主体的颜色和低频光影则近似拟合目标效果，需要视觉对照，不能保证每个主体像素相同。
- SIFT/RANSAC 的共享特征不足时，`transfer_enabled` 为 false。此时文件只是生成底图的原图尺寸版本，应说明没有可靠的真实细节可补，不降低门槛硬补。
- `reliable_fraction` 是受保护前景中局部映射权重 >0.25 的比例，是匹配覆盖率，不是通用画质评分。`structure_channel` 记录实际使用的结构表示，`fit_error_mean` 只衡量小图的局部拟合误差，均不能单独证明成片质量。
- 纵横比差异超过 2% 会停止，避免把原图拉伸来适配裁切后的底图。先检查生成是否遵守构图要求。
- 背景像素一致性以放大并转换到 sRGB 的生成底图为参照；透明度也以该底图为准。JPEG 是有损副本，不能用于逐像素一致性断言。
- 画布放大不能恢复背景或未匹配部位的真实细节；主体明显变形、换衣或图案重绘的部分应排除，保留用户有意的改动。

## 离线自检

`<python> <skill>/scripts/self_test.py --out <temporary-test-directory>` 使用合成像素与已知的高分辨率编辑目标，检查细节误差是否降低、低频颜色和光影是否保留，并测试强红光不引入灰/绿/蓝颗粒、有意修改区域排除、透明度保留、无对应内容时不传入细节，以及配置变化后拒绝复用旧配准。无需图片生成调用或网络。测试通过不替代每次实际图片的遮罩与局部对照检查。

局部模型与参数上采样思路参考 [Guided Image Filtering](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf)。这里实现的是受结构门槛约束的局部 RGB 曲线传递，不是对论文算法或 MIND 的完整复现。
