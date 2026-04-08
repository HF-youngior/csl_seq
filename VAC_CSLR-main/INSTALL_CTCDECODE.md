# ctcdecode installation notes (VAC_CSLR-main)

This project can train with `decode_mode=max` even if `ctcdecode` is missing.
To enable beam decoding (`decode_mode=beam`), install `ctcdecode` first.

## Preferred path: WayenVan/ctcdecode

```bash
cd /root/autodl-tmp/SLR
git clone https://github.com/WayenVan/ctcdecode.git
cd ctcdecode
git submodule update --init --recursive

# Important: disable isolated build so setup.py can import torch
MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir .
python -c "import ctcdecode; print('ctcdecode ok')"
```

## Fallback path

If build still fails, follow the compile flags workflow from:

`https://blog.csdn.net/m0_58341463/article/details/144365981`

Key checks:

1. `python -c "import torch"` must work before installing `ctcdecode`.
2. Keep `--no-build-isolation` enabled.
3. Ensure `gcc`, `g++`, and `cmake` are available.
4. If strict compiler flags fail, adjust flags/compiler version based on build log.

## Runtime verification

```bash
cd /root/autodl-tmp/SLR/VAC_CSLR-main
python -c "from utils.decode import Decode; print('decode import ok')"
```

Expected training startup logs:

- Beam path: `ctcdecode import ok. search_mode=beam ...`
- Fallback path: `ctcdecode import failed ... Fallback to greedy max decode`
