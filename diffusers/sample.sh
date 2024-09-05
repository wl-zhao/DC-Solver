# PYTHONPATH=./src:$PYTHONPATH \
#     python scripts/sample_dc_solver.py --mode search --NFE 5 --CFG 7.5
PYTHONPATH=./src:$PYTHONPATH \
    python scripts/sample_dc_solver.py --mode sample --NFE 20 --CFG 4.5 --use_cpr --logdir logs/cpr_nfe10_cfg7.5 --prompt "A high-fidelity photo of two big fierce tigers with big scary fangs running in the forest."
