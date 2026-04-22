# BCGModelling

Primary project documentation lives in `PROJECT_GUIDE.md`.

## Environment Recreation

- `environment.yml`: full lock-style export for reproducing this exact environment on similar systems.
- `environment.min.yml`: lightweight, portable spec exported from conda history.

Use one of:

```bash
conda env create -f environment.yml
# or
conda env create -f environment.min.yml
```
