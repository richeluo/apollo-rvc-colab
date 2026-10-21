# Apollo — Colab 一键 AI 翻唱管线

在 **Google Colab 免费版** 上，一格跑完一首 AI 翻唱：下载歌曲 → 分离人声/伴奏 → 用你自己训练的 RVC 模型只转人声 → 响度归一 → 与原伴奏混音出成品。

基于 [Applio](https://github.com/IAHispano/Applio) 推理层 + [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（UVR）分离。实测：一首 3 分半的歌，在免费 T4 上新会话全程约 6 分钟。

## 特色

- **真·一键**：依赖（`torchcrepe`、`faiss-cpu` 等）自动安装，不用手动补包
- **模型自动下载**：把训练好的 `.pth` + `.index` 放到 HuggingFace 仓库（私有也可以），新会话自动拉取，告别手动拖文件
- **只转干声**：伴奏保持原样，成品音量/伴奏稳定，不会出现整首歌被"转糊"的问题
- **参数可调**：`--pitch` 升降调、`--index-rate` 音色强度、`--protect` 辅音保护

## 快速开始（Colab）

**第 1 步**：打开 Applio 官方 notebook（`Applio_NoUI.ipynb`），切到 GPU 运行时（T4），运行 **Setup runtime environment** 那一格。

**第 2 步**（可选，模型放私有 HF 仓库时才需要）：在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 建一个 Read 令牌；Colab 左侧 🔑 Secrets 里加一条 `HF_TOKEN`，打开 Notebook access。

**第 3 步**：新开一格，粘贴运行（模型在公开仓库或已手动上传时，可省去前三行取令牌代码）：

```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')   # userdata 只有格子进程能读，必须经环境变量传递

!wget -q "https://raw.githubusercontent.com/<你的用户名>/apollo-rvc-colab/main/apollo_pipeline.py" -O /content/apollo_pipeline.py
!python /content/apollo_pipeline.py --song <歌曲音频直链> --model-name <模型名> --model-repo <用户名>/<模型仓库>
```

成品 `apollo_cover.mp3` / `.wav` 会出现在 Colab 根目录，左侧文件面板下载即可（挂了 Drive 的话还会同步一份）。

## 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--song` | 歌曲音频直链（mp3/wav） | 必填 |
| `--model-name` | 模型名，对应 `<名>.pth` + `<名>.index` | `MyVoice` |
| `--model-repo` | 托管模型的 HuggingFace 仓库 | 空（仅本地查找） |
| `--pitch` | 升降调（半音），如 `2` / `-2` | `0` |
| `--index-rate` | 音色相似度强度 | `0.75` |
| `--protect` | 辅音/气声保护 | `0.33` |
| `--fetch-model` | 只测模型下载、随即退出 | — |

## 前置条件

- 你自己的 RVC 模型（用 Applio/RVC 训练，10–30 分钟干净清唱干声起步）。管线推理参数与训练配置对齐：F0 `rmvpe`、embedder `chinese-hubert-base`，训练时请保持一致。
- 歌曲给**直链**（能直接下载到音频文件的 URL），不是分享页链接。

## English

One-cell AI cover-song pipeline for free-tier Google Colab, built on Applio + UVR separation. Downloads your song, separates vocals, converts **only the vocals** with your own trained RVC model, loudness-normalizes, and mixes them back over the original instrumental. Models can be auto-fetched from a (private) HuggingFace repo — no manual file uploads per session. A ~3.5-minute song renders end-to-end in ~6 minutes on a free T4. See the quick start above.

## 注意 / Notes

- 请只使用你本人或已获授权的声音模型与歌曲，翻唱作品的发布请遵守当地版权规定。
- Voice models and songs you use must be your own or properly licensed.

## License

MIT
