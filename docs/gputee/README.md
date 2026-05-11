# BCGModelling — gputee docs

Active documentation for the current host: `gputee` (1× NVIDIA H100 PCIe, 80 GB VRAM).

- `PROJECT_GUIDE.md` — single source of truth for the pipeline, data, scripts, and status.
- `FINETUNE_GUIDE.md` — Evo2 7B fine-tuning: hardware, hyperparameters, launch commands, logging.
- `BGC_Research_Plan.md` — full research plan.
- `MIGRATION_CHANGELOG.md` — every change made when porting from the previous
  `trojai` host (4× NVIDIA A40) to `gputee`, with rationale. Read this to
  understand the delta from the archived `docs/trojai/` copy.

## Environment Recreation

`environment.yml` (full lock) and `environment.min.yml` (portable) live at
the repo root.

gputee has micromamba, not conda:

```bash
micromamba create -n bgcmodel -f environment.yml
micromamba activate bgcmodel
```

> **`environment.yml` alone does not produce a working env on a fresh
> create.** The pip step crashes on `flash-attn` because `torch` is not
> yet installed when `flash-attn`'s `setup.py` runs `import torch`. The
> conda side does finish cleanly. See **`FINETUNE_GUIDE.md` §2** for
> the working install sequence (torch first, prebuilt flash-attn wheel,
> then re-run `env update`, then deepspeed/peft/wandb).

On conda-equipped hosts the equivalent is `conda env create -f environment.yml`
(same caveat applies).
