# SignMatch

Code for [SignMatch: Matching Dictionary Signs to Continuous Sign Language
Video](https://arxiv.org/abs/2609.01886).

## Install

```bash
pip install -r requirements.txt
```

## Weights

- SignMatch: [`signmatch_rgb_v1.pt`](https://github.com/ryanwongsa/signmatch/releases/tag/v1.0.0)
- SignRep (for ASL-Citizen trimming and training): `ckpt.pt` from the
  [SignRep `v0.0.1` release](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1)

## Inference

Input videos should be signer-cropped and 25 fps.

```bash
python scripts/embed.py dictionary dictionary/*.mp4 --checkpoint signmatch_rgb_v1.pt -o dictionary.pt
python scripts/embed.py continuous signing.mp4 --checkpoint signmatch_rgb_v1.pt -o signing.pt
python scripts/match.py --queries signing.pt --dictionary dictionary.pt --top-k 5
```

From Python, run from the repository root:

```python
from inference import (
    embed_continuous, embed_dictionary, iter_continuous_video, load_dictionary_video, load_model, match,
)

model = load_model("signmatch_rgb_v1.pt", device="cuda")
house = embed_dictionary(model, load_dictionary_video("dictionary/house.mp4"))
for windows, spans in iter_continuous_video("signing.mp4"):
    scores = match(embed_continuous(model, windows), house[None])[:, 0]
    print(spans[scores.argmax()], scores.max().item())
```

## ASL-Citizen

Download [ASL-Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) to
`data/ASL_Citizen` (`videos/` and `splits/`), then trim and score:

```bash
python scripts/prepare_aslcitizen.py --raw data/ASL_Citizen/videos --signrep ckpt.pt --out data/ASL_Citizen/clips --gpu 0
python scripts/eval_weights.py --weights signmatch_rgb_v1.pt --clips data/ASL_Citizen/clips --out results/aslcitizen --gpu 0
```

## Training data

Training reads from `data/` (or `$DATA_ROOT`):

```
data/
  signrep.pt                 SignRep ckpt.pt
  lmdbs/<name>/              video frames
  spottings/{train,val,test}.csv
  dictionary.csv             training dictionary
  val_dictionary.csv         dictionary that val and test spottings are ranked against
```

Frames are 256x256 JPEGs at 25 fps, one LMDB per `<name>`, keyed `<video>_<i>` for frame `i`, with
`<video>_details` holding `pickle.dumps({"num_frames": N})`.

Spottings (frames inclusive; `label` is a dictionary `id`, `lmdb` a folder in `lmdbs/`):

```csv
video_id,start_frame,end_frame,label,lmdb
signer01_conversation,112,131,HOUSE,continuous
```

Dictionaries (one video per `id`; the signing span is inclusive):

```csv
id,lmdb,video_name,num_frames,first_active,last_active
HOUSE,dictionary,house_01,61,9,52
```

## Training

```bash
gatle-ignite train --config=configs/stage1.py
gatle-ignite eval --config=configs/stage1.py --ckpt best
gatle-ignite train --config=configs/stage2.py
gatle-ignite eval --config=configs/stage2.py --ckpt best
python scripts/export_weights.py --src checkpoints/stage2/valid_best_result_checkpoint_<epoch>_<score>.pt --out my_weights.pt
```

`my_weights.pt` is used like `signmatch_rgb_v1.pt` above.

## Citation

```bibtex
@inproceedings{wong2026signmatch,
  title     = {SignMatch: Matching Dictionary Signs to Continuous Sign Language Video},
  author    = {Wong, Ryan and Jang, Youngjoon and Momeni, Liliane and Varol, G{\"u}l and Zisserman, Andrew},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## Licence

Non-commercial use only: see [LICENSE](LICENSE). The Video-Hiera code in `models/model_utils/hiera/`
is Meta's, under its own [licence](models/model_utils/hiera/LICENSE).
