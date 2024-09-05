<!--Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# CogVideoX

[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) from Tsinghua University & ZhipuAI, by Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, Jie Tang.

The abstract from the paper is:

*We introduce CogVideoX, a large-scale diffusion transformer model designed for generating videos based on text prompts. To efficently model video data, we propose to levearge a 3D Variational Autoencoder (VAE) to compresses videos along both spatial and temporal dimensions. To improve the text-video alignment, we propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between the two modalities. By employing a progressive training technique, CogVideoX is adept at producing coherent, long-duration videos characterized by significant motion. In addition, we develop an effectively text-video data processing pipeline that includes various data preprocessing strategies and a video captioning method. It significantly helps enhance the performance of CogVideoX, improving both generation quality and semantic alignment. Results show that CogVideoX demonstrates state-of-the-art performance across both multiple machine metrics and human evaluations. The model weight of CogVideoX-2B is publicly available at https://github.com/THUDM/CogVideo.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

There are two models available that can be used with the CogVideoX pipeline:
- [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b)
- [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b)

## Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b").to("cuda")
```

Then change the memory layout of the pipelines `transformer` component to `torch.channels_last`:

```python
pipe.transformer.to(memory_format=torch.channels_last)
```

Finally, compile the components and run inference:

```python
pipe.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

# CogVideoX works well with long and well-described prompts
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
```

The [benchmark](https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f) results on an 80GB A100 machine are:

```
Without torch.compile(): Average inference time: 96.89 seconds.
With torch.compile(): Average inference time: 76.27 seconds.
```

### Memory optimization

CogVideoX-2b requires about 19 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/a-r-r-o-w/3959a03f15be5c9bd1fe545b09dfcc93) script.

- `pipe.enable_model_cpu_offload()`:
  - Without enabling cpu offloading, memory usage is `33 GB`
  - With enabling cpu offloading, memory usage is `19 GB`
- `pipe.enable_sequential_cpu_offload()`:
  - Similar to `enable_model_cpu_offload` but can significantly reduce memory usage at the cost of slow inference
  - When enabled, memory usage is under `4 GB`
- `pipe.vae.enable_tiling()`:
  - With enabling cpu offloading and tiling, memory usage is `11 GB`
- `pipe.vae.enable_slicing()`

### Quantized inference

[torchao](https://github.com/pytorch/ao) and [optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be used to quantize the text encoder, transformer and VAE modules to lower the memory requirements. This makes it possible to run the model on a free-tier T4 Colab or lower VRAM GPUs!

It is also worth noting that torchao quantization is fully compatible with [torch.compile](/optimization/torch2.0#torchcompile), which allows for much faster inference speed. Additionally, models can be serialized and stored in a quantized datatype to save disk space with torchao. Find examples and benchmarks in the gists below.
- [torchao](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897)
- [quanto](https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa)

## CogVideoXPipeline

[[autodoc]] CogVideoXPipeline
  - all
  - __call__

## CogVideoXVideoToVideoPipeline

[[autodoc]] CogVideoXVideoToVideoPipeline
  - all
  - __call__

## CogVideoXPipelineOutput

[[autodoc]] pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput
