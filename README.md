# BCGModelling

Fine-tune Evo2 7B to generate synthesis-ready biosynthetic gene cluster (BGC)
nucleotide sequences conditioned on compound class, compound identity, and
taxonomic lineage.

## Documentation

The project docs are versioned per host, because hardware-specific guidance
(GPU count, memory budgets, launch commands) varies by server:

- `docs/gputee/` — **active** docs for the current host (1× NVIDIA H100 PCIe, 80 GB).
  Start here.
- `docs/trojai/` — archived docs for the previous host (4× NVIDIA A40, 48 GB each).
  Kept unchanged for historical reference.

Within each folder:

- `PROJECT_GUIDE.md` — single source of truth for the pipeline (data, scripts, metrics, status).
- `FINETUNE_GUIDE.md` — Evo2 7B fine-tuning: hardware, hyperparameters, logging, checkpointing.
- `BGC_Research_Plan.md` — full research plan.
- `README.md` — short local entry point.

`docs/gputee/MIGRATION_CHANGELOG.md` records every change made when porting
the project from trojai to gputee and why.

## Environment recreation

- `environment.yml` — full lock-style export (conda).
- `environment.min.yml` — portable spec from conda history.

```bash
conda env create -f environment.yml
# or, if conda is not installed (e.g. on gputee):
micromamba create -n bgcmodel -f environment.yml
```

See `docs/gputee/PROJECT_GUIDE.md` §3 for the full GPU-stack install sequence.
