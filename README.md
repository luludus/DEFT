## DEFT: Dynamic Entropy Fine-Tuning

### Abstract

Standard negative log-likelihood (NLL) for Supervised Fine-Tuning (SFT) applies uniform token-level weighting. This rigidity creates a two-fold failure mode: (i) overemphasizing low-probability targets can amplify gradients on noisy supervision and disrupt robust priors, and (ii) uniform weighting provides weak sharpening when the model is already confident.
Existing methods fail to resolve the resulting plasticity--stability dilemma, often suppressing necessary learning signals alongside harmful ones.
To address this issue, we unify token-level SFT objectives within a generalized deformed-log family and expose a universal gate x error gradient structure, where the gate controls how much the model trusts its current prediction.
By employing the Cayley transform, we map the model's continuously evolving uncertainty onto a continuous focus trajectory, which enables seamless interpolation between scenarios involving uncertain novel concepts and those involving well-established knowledge.
We then introduce **Dynamic Entropy Fine-Tuning (DEFT)**, a parameter-free objective that modulates the trust gate using distribution concentration (Rényi-2 entropy) as a practical proxy for the model's predictive state. Extensive experiments and analyses demonstrate that DEFT achieves a better balance between exploration and exploitation, leading to improved overall performance.

### Repository layout

- `main_verl/trainer/fsdp_sft_trainer.py`: FSDP SFT trainer with token-level objective transformations.
- `main_verl/trainer/config/sft_trainer.yaml`: Default Hydra config for SFT.
- `scripts/one_click/`: One-command script generator to run training + evaluation.
- `scripts/training/` and `scripts/evaluation/`: Example scripts per dataset.
- `data/`: Data preparation scripts.
- `evaluations/`: Evaluation code for Math / Medical / FigFont.

### Requirements

This repo assumes a modern PyTorch + Transformers training stack.

- **Training (core)**
  - Python 3.10+
  - `transformers`
  - `verl(==0.4.0.dev0)`
  - `torch`
  - `vllm`
  - `flash_attn`

- **Evaluation (Math)**
  - See `evaluations/math/requirements.txt` and `evaluations/math/latex2sympy` (includes `word2number`, `latex2sympy2`, etc.)

### Quick start


```bash
python scripts/one_click/script_generator.py \
  --dataset $DATASET \
  --model_save_name $MODEL_KEY \
  --trainer_objective_trans $OBJECTIVE \
  (--run_script)
```

- **`$DATASET`**: `math` | `medical` | `figfont`
- **`$MODEL_KEY`**: one of the predefined keys in `scripts/one_click/TEMPLATE.py` (e.g., `qwen-2.5-math-7b`, `llama-3.1-8b`, ...)
- **`$OBJECTIVE`**: the token-level SFT objective transformation, passed to `trainer.objective_trans`
- **`--run_script`**: optional; Boolean flag if set, the generated script will be executed immediately
- **`nproc_per_node`**: (Optional) Specifies the number of GPUs to use.
- **`cuda_visible_devices`**: (Optional) Specifies specific GPU devices (e.g., --cuda_visible_devices 0,1,2,3).


### Objective transformations

The trainer supports multiple token-level objective transformations.

Common examples:

- **Original SFT / NLL**: `original`
- **Probability error**: `p` 
- **Cayley transform focus trajectory**: `Cayley_Trans`
- **Dynamic Entropy Fine-Tuning**: `DEFT`


### Acknowledgements
The implementation of this repository is built upon [veRL](https://github.com/volcengine/verl) and [Beyond-Log-Likelihood](https://github.com/GaotangLi/Beyond-Log-Likelihood). We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

### Citation
If you find this repository useful, please cite:
```bash
@misc{wang2026gradientsearninfluenceunifying,
      title={Gradients Must Earn Their Influence: Unifying SFT with Generalized Entropic Objectives}, 
      author={Zecheng Wang and Deyuan Liu and Chunshan Li and Yupeng Zhang and Zhengyun Zhao and Dianhui Chu and Bingning Wang and Dianbo Sui},
      year={2026},
      eprint={2602.11424},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11424}, 
}
```
