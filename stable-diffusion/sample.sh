PYTHONPATH=./:$PYTHONPATH \
python scripts/sample_diffusion_dc_solver.py -r checkpoints/ffhq256/model.ckpt --mode search -c 5
PYTHONPATH=./:$PYTHONPATH \
python scripts/sample_diffusion_dc_solver.py -r checkpoints/ffhq256/model.ckpt --mode sample -c 5 --logdir logs/ffhq