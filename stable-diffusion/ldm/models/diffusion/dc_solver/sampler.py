"""SAMPLING ONLY."""

import torch

from .dc_solver import NoiseScheduleVP, model_wrapper, DCSolver
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
import json
import os
import copy

class DCSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))
        self.ddim_sampler = DDIMSampler(model)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               order=3,
               mode='sample',
               ddim_gt_path=None,
               dc_ratios=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        if mode == 'search':
            assert ddim_gt_path is not None
            dc_solver = DCSolver(model_fn, ns, predict_x0=True, thresholding=False)
            if not os.path.isfile(ddim_gt_path):
                os.makedirs(os.path.dirname(ddim_gt_path), exist_ok=True)
                _, intermediates = self.ddim_sampler.sample(S=200, batch_size=batch_size, shape=shape, eta=0, x_T=x_T, log_every_t=1)
                torch.save({
                    'xs': intermediates['x_inter'][:-1],
                    'ts': np.array(intermediates['ts']) / dc_solver.noise_schedule.total_N,
                }, ddim_gt_path)
            ddim_gt = torch.load(ddim_gt_path)
            dc_solver.ref_ts = ddim_gt['ts']
            dc_solver.ref_xs = ddim_gt['xs']
            dc_ratios = dc_solver.search_dc(img, steps=S, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True)
            return dc_ratios
        elif mode == 'sample':
            assert dc_ratios is not None
            dc_solver = DCSolver(model_fn, ns, predict_x0=True, thresholding=False)
            x = dc_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True, dc_ratios=copy.deepcopy(dc_ratios))
            return x.to(device)
