# PVQD speaker-verification script

`pvqd.py` is the editable pipeline. It works as `import pvqd` or as a command-line script.
Importing it does not install packages, download files, create directories, or load a model.
Downloads are restricted to hosted Colab. The existing notebook is unchanged.

## Start on Colab

Choose a GPU runtime. Clone your repository, then install the separate Colab requirements:

```python
!git clone https://github.com/YOUR_USER/YOUR_REPO.git /content/pvqd_code
%cd /content/pvqd_code
%pip install -q -r requirements/pvqd_colab.txt
```

Restart the Colab session after installing. Then:

```python
%cd /content/pvqd_code
import pvqd

cfg = pvqd.Config(root="/content/pvqd_voiceprivacy")
clips = pvqd.prepare(cfg)
pvqd.download_model(cfg)
model = pvqd.load_model(cfg)
```

All large downloads, snippets, checkpoints, and results go under `cfg.root` on Colab.
The data and model download paths match the earlier notebook, so existing files can be reused.
This script uses the checked-in direct-download listings rather than the Mendeley listing API
that returned 403. If a direct file download itself fails, its error is surfaced.

For a small new repository, the required files are:

```text
pvqd.py
requirements/pvqd_colab.txt
datasets/pvqd/metadata/audio_files.json
datasets/pvqd/metadata/rating_files.json
```

The tests and this guide are useful additions. Do not commit audio, model weights, snippets, or
training outputs. PVQD is [CC BY 4.0, by Patrick R. Walden / The Voice Foundation](https://data.mendeley.com/datasets/9dz247gnyb/4).
The downloaded VoicePrivacy source includes its license.

## Run individual components

Inspect a pretrained **development** result before training:

```python
scores = pvqd.score_trials(model, clips, split="dev", batch_size=cfg.batch_size)
print("Pretrained development EER:", 100 * pvqd.eer(scores))
print(scores.groupby("target").score.agg(["mean", "std", "min", "max"]))
```

Fine-tune, select the checkpoint using development EER, then evaluate both models on test:

```python
model, history = pvqd.train(cfg, clips, model)
results = pvqd.evaluate(cfg, clips, model, split="test")
```

Other useful entry points:

| Call | Returns / effect |
| --- | --- |
| `pvqd.read_metadata(cfg)` | Audio listing and GRBAS workbook record; no network |
| `pvqd.download_data(cfg)` | Download audio and workbook on Colab |
| `pvqd.make_snippets(cfg)` | Clip dataframe from cached audio; no split or trial roles |
| `pvqd.split_speakers(clips, seed=2026)` | Copy with speaker-disjoint train/dev/test assignments |
| `pvqd.assign_roles(clips, seed=2026)` | Copy with shuffled enrollment/probe roles per speaker |
| `pvqd.validate_clips(clips)` | Reject overlap and speaker/source leakage |
| `pvqd.read_clips(cfg)` | Reload saved `results/snippets.csv` |
| `pvqd.sample_epoch(clips, seed=2026)` | Shuffled, speaker-balanced training rows |
| `pvqd.extract_embeddings(model, frame, batch_size=4)` | Normalized embeddings, in dataframe row order |
| `pvqd.cosine_trials(frame, vectors)` | Trial scores from existing row-aligned embeddings |
| `pvqd.score_trials(model, clips, split="dev")` | Trial dataframe; no hidden global clip table |
| `pvqd.eer(scores)` | EER fraction; multiply by 100 for percent |
| `pvqd.load_embedding(model, checkpoint_path)` | Replace ECAPA weights without reloading WavLM |
| `pvqd.train(cfg, clips)` | Fresh pretrained model, then `(best_model, history_dataframe)` |
| `pvqd.evaluate(cfg, clips, model)` | EER dataframe; saves per-trial scores and summary |

If you only want a particular checkpoint evaluated:

```python
results = pvqd.evaluate(
    cfg, clips, model, split="dev",
    checkpoints={"pretrained": cfg.models / "embedding_model.ckpt"},
)
```

To reuse the snippet manifest from the old notebook without reprocessing audio:

```python
import pandas as pd

clips = pd.read_csv(
    "/content/pvqd_voiceprivacy/lean_results/snippets.csv",
    dtype={"speaker": str, "source_hash": str},
)
clips = pvqd.assign_roles(clips, seed=cfg.seed)
pvqd.validate_clips(clips)
```

## Edit and reload

Edit `pvqd.py` directly in Colab's file editor or update your clone. Reload it before calling an
edited function:

```python
import importlib
importlib.reload(pvqd)
```

If you edit model construction or `forward`, call `pvqd.load_model(cfg)` again as well. Changing
`cfg.seed` or `cfg.seconds` alone does not modify an existing dataframe: rerun the relevant
preprocessing function. When changing trial roles, use the same resulting dataframe for both models.

## What changed from the notebook

- The speaker split remains 60/20/20, stratified by severity. Speakers and source recordings cannot
  cross splits, and snippet windows cannot overlap.
- Enrollment and probe clips are now selected by a reproducible shuffle **within each speaker**,
  instead of early clips versus late clips. Two enroll, two probe, and up to two additional clips
  are available for training. The shuffle is independent of dataframe row order.
- This is a **mixed-task** evaluation: shuffling removes the systematic sentence-to-vowel ordering
  bias, but does not guarantee matched tasks. For sentence-only or vowel-only trials, annotate
  and filter the clip table first, then assign roles using enough disjoint clips per speaker.
- Default sampling gives every training speaker the same number of examples. Severity stratifies
  the split; it is not a proxy for identity. Set `cfg.balance = "severity"` to equalize severity-group
  totals instead. A severity-balanced group may repeat its speakers more often.
- WavLM remains frozen. ECAPA uses learning rate `1e-5`; the new speaker classifier uses `1e-3`.
  These are editable starting values, not a validated optimum. BatchNorm running statistics remain
  frozen during training, while affine parameters train. Non-finite loss, gradients, or embeddings
  stop execution instead of being replaced with zeros or dropped from EER.
- Training prints loss, training-speaker classification accuracy, and development EER. The best
  checkpoint uses development EER; ties retain the earlier epoch, which is printed explicitly.
  `train` returns the selected model. It starts fresh if no model is supplied; it is not a resume API.
- Test results should be used after choosing the protocol/settings on development speakers.

Default outputs are in `/content/pvqd_voiceprivacy/results`: the snippet manifest, exclusions,
configuration, training history, `best_finetuned.pt`, per-trial score CSVs, and EER tables.
Repeating a run with the same root overwrites those results. The embedding checkpoint must be
loaded together with the original frozen WavLM weights.

No anonymization is applied. Lower EER means a stronger verifier on these trials. PVQD clips from
the same recording measure within-recording recognition, not recognition across visits. This is a
PVQD adaptation of the VoicePrivacy attacker, not an official challenge evaluation score.

## Command line

From the Colab clone:

```bash
python pvqd.py prepare
python pvqd.py download-model
python pvqd.py train --epochs 3 --lr 1e-5 --classifier-lr 1e-3
python pvqd.py evaluate
```

Or run all steps with `python pvqd.py run`. Use `python pvqd.py --help` for settings. These commands
run the same functions shown above; dependency installation is an explicit Colab step.

CPU-only checks, with the data/scoring dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_pvqd_module.py' -v
```

Tests do not download assets or require a GPU. Tests whose numerical dependencies are unavailable
are skipped; GPU fine-tuning still needs to be checked in Colab.
