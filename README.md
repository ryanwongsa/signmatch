# SignMatch: Matching Dictionary Signs to Continuous Sign Language Video

Official Codebase for: https://arxiv.org/abs/2609.01886v1

Match sign videos by visual similarity. Given a dictionary of isolated sign videos, it answers: does this sign appear in this video, and where?

Currently Inference code for "SignMatch: Matching Dictionary Signs to Continuous Sign Language Video". 
Work in progress to release the keypoint model and pipelines.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements-gpu.txt   # CUDA
pip install -r requirements.txt       # CPU only
```


## Checkpoint

Add the checkpoint from the release tab of this repo:
```
checkpoints/signmatch_rgb.pt 
```



## Use

```bash
# 1. embed a dictionary
python scripts/embed.py dictionary dictionary/*.mp4 -o dict.pt

# 2. embed the video to search
python scripts/embed.py continuous signing.mp4 -o segments.pt

# 3. match
python scripts/match.py --queries segments.pt --dictionary dict.pt --top-k 5
```

---

## License

This software is licensed under the SignMatch Non-Commercial License. See [LICENSE](LICENSE).

Documentation, text, and other non-code materials in this repository are licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). To view a copy of the license, visit https://creativecommons.org/licenses/by-nc/4.0/

> Copyright (c) 2026, SignMatch contributors. All rights reserved.

