import argparse
import os
import torch
import json

from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd', default='2.1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', default='search', type=str, choices=['search', 'sample'])
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--CFG', default=7.5, type=float, help='classifier-free guidance scale')
    parser.add_argument('--NFE', default=10, type=int, help='number of function evaluations')
    parser.add_argument('--dc_ratios_path', default='dc_solver/dc_ratios/default.json', type=str)
    parser.add_argument('--ddim_gt_path', default=None, type=str)
    parser.add_argument('--cpr_coeffs_path', default='dc_solver/cpr_coeffs_path/sd2.1.npy', type=str)
    parser.add_argument('--use_cpr', default=False, action='store_true')
    parser.add_argument('--prompt', default='A photo of a serene coastal cliff with waves crashing against the rocks below.', type=str)
    parser.add_argument('--logdir', default='logs/samples', type=str)

    args = parser.parse_args()
    if args.ddim_gt_path is None:
        args.ddim_gt_path = f'dc_solver/ddim_gt/sd{args.sd}_CFG_{args.CFG}.pth'
    return args

args = parse_args()
model_id_dict = {
    '1.5': 'runwayml/stable-diffusion-v1-5',
    '1.4': 'CompVis/stable-diffusion-v1-4',
    '2.1': 'stabilityai/stable-diffusion-2-1',
}
shape_dict = {
    '1.4': [args.batch_size, 4, 64, 64],
    '1.5': [args.batch_size, 4, 64, 64],
    '2.1': [args.batch_size, 4, 96, 96],
}

extra_kwargs = {}
if args.sd == '2.1':
    extra_kwargs = {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "prediction_type": "v_prediction",
    }
model_id = model_id_dict[args.sd]

scheduler_kwargs = dict(
    beta_start=0.00085,
    beta_end=0.012,
    solver_order=2,
    prediction_type="epsilon",
    thresholding=False,
    solver_type='bh2',
    lower_order_final=True,
    dc_order=2,
    disable_corrector=[0],
)
scheduler_kwargs.update(**extra_kwargs)

def generate_gt(args, noise, prompts):
    def get_callback(t_list, latent_list):
        def callback(step_index, t, latents):
            t_list.append(t)
            latent_list.append(latents)
        return callback
    t_list = []
    latent_list = []
    pipe = StableDiffusionPipeline.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to(args.device)
    callback = get_callback(t_list, latent_list)
    _imgs = pipe(prompts, num_inference_steps=200, output_type="latent", latents=noise, callback=callback, guidance_scale=args.CFG, callback_steps=1).images
    t_tensor = torch.stack(t_list)
    latent_tensor = torch.stack(latent_list, dim=1) # B, T, C, H, W
    os.makedirs(os.path.dirname(args.ddim_gt_path), exist_ok=True)
    ddim_gt = {
        'ts': t_tensor,
        'intermediates': latent_tensor,
    }
    torch.save(ddim_gt, args.ddim_gt_path)
    return ddim_gt

def search(args):
    shape_str = 'x'.join(f'{s}' for s in shape_dict[args.sd])
    noise_path = f'noises/{shape_str}.pth'
    if os.path.isfile(noise_path):
        noise = torch.load(noise_path, map_location=args.device)
    else:
        noise = torch.randn(shape_dict[args.sd], device=args.device, dtype=torch.float16)
        os.makedirs(os.path.dirname(noise_path), exist_ok=True)
        torch.save(noise, noise_path)
    
    with open('captions.json') as f:
        captions = json.load(f)

    prompts = [item['caption'] for item in captions]

    if not os.path.isfile(args.ddim_gt_path):
        print('ddim gt does not exist, generate for once')
        ddim_gt = generate_gt(args, noise, prompts)
    else:
        ddim_gt = torch.load(args.ddim_gt_path, map_location=args.device)
        
        if ddim_gt['intermediates'].shape[0] > args.batch_size:
            ddim_gt['intermediates'] = ddim_gt['intermediates'][:args.batch_size]
            prompts = prompts[:args.batch_size]
            
    from diffusers.schedulers.scheduling_dcsolver_multistep import DCSolverMultistepScheduler
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_dcsolver import StableDiffusionPipelineDCSolver
    dcsolver = DCSolverMultistepScheduler(ddim_gt=ddim_gt, **scheduler_kwargs)
    pipe = StableDiffusionPipelineDCSolver.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = dcsolver
    pipe = pipe.to(args.device)
    _ = pipe(prompts, num_inference_steps=args.NFE, output_type="latent", latents=noise, guidance_scale=args.CFG)
    dc_ratios = dcsolver.dc_ratios
    os.makedirs(os.path.dirname(args.dc_ratios_path), exist_ok=True)
    json.dump(dc_ratios, open(args.dc_ratios_path, 'w'))

def sample(args):
    from diffusers.schedulers.scheduling_dcsolver_multistep import DCSolverMultistepScheduler
    dcsolver = DCSolverMultistepScheduler(**scheduler_kwargs)
    if args.use_cpr:
        dc_ratios = dcsolver.cascade_polynomial_regression(args.CFG, args.NFE, args.cpr_coeffs_path)
    else:
        dc_ratios = json.load(open(args.dc_ratios_path))

    dcsolver.dc_ratios = dc_ratios
    pipe = StableDiffusionPipeline.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to(args.device)
    pipe.scheduler = dcsolver
    image = pipe(args.prompt, num_inference_steps=args.NFE, guidance_scale=args.CFG).images[0]
    os.makedirs(args.logdir, exist_ok=True)
    num_images = len(os.listdir(args.logdir))
    image.save(f'{args.logdir}/img_{num_images:05d}.png')


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'search':
        search(args)
    elif args.mode == 'sample':
        sample(args)