# DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation
This code contains the Pytorch implementation for DC-Solver.

## Installation
We implement our DC-Solver in both [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) and [diffusers](https://github.com/huggingface/diffusers), please follow their instructions to install the environment. Our implementation of DC-Solver for the two codebases are in the `stable-diffusion` and `diffusers` folders, respectively. Please clone their repo and copy our code of DC-Solver to the corresponding codebase.

## Usage
### Stable-Diffusion
We take unconditional sampling on FFHQ as example. Please first prepare the checkpoint on FFHQ using the link provided [here](https://github.com/CompVis/stable-diffusion/blob/main/scripts/download_models.sh). We can then perform the search stage to find the optim dynamic compensation ratios by:
```
PYTHONPATH=./:$PYTHONPATH \
    python scripts/sample_diffusion_dc_solver.py -r models/ldm/ffhq256/model.ckpt --mode search --custom_steps 5
```
Note that we need to generate the ground truth trajectory using DDIM before the first search. This is automatically done by the above command in the first run and will not cost extra computational costs later.

The dynamic compensation ratios are stored in `dc_solver/dc_ratios/default.json` by default. We can then run the sampling stage by:
```
PYTHONPATH=./:$PYTHONPATH \
python scripts/sample_diffusion_dc_solver.py -r models/ldm/ffhq256/model.ckpt --mode sample --custom_steps 5
```

### Diffusers
We perform experiments of text-to-image generation using diffusers library. The search stage can be performed by:
```
PYTHONPATH=./src:$PYTHONPATH \
    python scripts/sample_dc_solver.py --mode search --NFE 5 --CFG 7.5
```
And we can then run the sampling via
```
PYTHONPATH=./src:$PYTHONPATH \
    python scripts/sample_dc_solver.py --mode sample --NFE 5 --CFG 7.5
```

We can also use the cascade polynomial regression to instantly predict the dynamic compensation ratios on unseen NFE/CFG by just passing the `--use_cpr` flag, for example:
```
PYTHONPATH=./src:$PYTHONPATH \
    python scripts/sample_dc_solver.py --mode sample --NFE 6 --CFG 6.0 --use_cpr
```

## Acknowledgement
Our code is based on [UniPC](https://github.com/wl-zhao/UniPC), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion), [diffusers](https://github.com/huggingface/diffusers). Thanks for their awesome work!
