"""PVQD original-audio speaker verification, as importable steps or a CLI.

Colab: ``import pvqd; cfg = pvqd.Config(); clips = pvqd.prepare(cfg)``.
Importing this file does not install packages, access the network, or load models.
Audio/model downloads are permitted only on a hosted Colab instance.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile


REVISION = "d5c7f4393db3e09247f58a56ca4d340fcf1ed850"
METADATA = Path(__file__).resolve().parent / "datasets/pvqd/metadata"
SAMPLE_RATE = 16000


@dataclass
class Config:
    root: str = "/content/pvqd_voiceprivacy"
    metadata_dir: str = str(METADATA)
    seed: int = 2026
    seconds: float = 3.0
    batch_size: int = 4
    epochs: int = 3
    lr: float = 1e-5
    classifier_lr: float = 1e-3
    examples_per_speaker: int = 6
    balance: str = "speaker"  # Or "severity"; labels are always speaker IDs.
    device: str = "cuda"

    @property
    def data(self):
        return Path(self.root) / "datasets/pvqd"

    @property
    def snippets(self):
        return Path(self.root) / f"snippets_{self.seconds:g}s"

    @property
    def models(self):
        return Path(self.root) / "models/voiceprivacy2026"

    @property
    def out(self):
        return Path(self.root) / "results"


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, path, sha256=None):
    """Download on Colab, caching files and checking publisher hashes when supplied."""
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Audio/model downloads run only on Colab.") from exc
    if not Path("/content").is_dir():
        raise RuntimeError("Use a hosted Colab runtime for downloads.")
    path = Path(path)
    if path.is_file() and (sha256 is None or sha256_file(path) == sha256):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as source, part.open("wb") as target:
                expected = int(source.headers.get("Content-Length", 0))
                shutil.copyfileobj(source, target)
            if expected and part.stat().st_size != expected:
                raise OSError(f"Incomplete download: {path.name}")
            if sha256 and sha256_file(part) != sha256:
                raise OSError(f"SHA-256 mismatch: {path.name}")
            part.replace(path)
            return path
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def read_metadata(cfg):
    """Read the two small, checked-in listings; no Mendeley API discovery request."""
    directory = Path(cfg.metadata_dir)
    audio = json.loads((directory / "audio_files.json").read_text())
    ratings = json.loads((directory / "rating_files.json").read_text())
    grade = next(r for r in ratings if r["filename"] == "grbas_grade_only.xlsx")
    return audio, grade


def download_data(cfg):
    """Fetch PVQD WAVs and its GRBAS workbook using the saved direct URLs."""
    from tqdm.auto import tqdm

    audio, grade = read_metadata(cfg)

    def fetch(record):
        name, info = record["filename"], record["content_details"]
        if Path(name).name != name:
            raise ValueError(f"Unsafe filename: {name}")
        folder = cfg.data / ("audio" if name.lower().endswith(".wav") else "metadata")
        return download_file(info["download_url"], folder / name, info["sha256_hash"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(tqdm(pool.map(fetch, audio + [grade]), total=len(audio) + 1, desc="PVQD"))


def normalize_id(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def read_severity(cfg):
    import pandas as pd

    grade = pd.read_excel(cfg.data / "metadata/grbas_grade_only.xlsx", header=None).iloc[1:]
    return {normalize_id(row.iloc[0]): str(row.iloc[11]).strip()
            for _, row in grade.iterrows() if pd.notna(row.iloc[0]) and pd.notna(row.iloc[11])}


def snippet_starts(audio, seconds=3.0):
    """Select at most six non-overlapping, non-silent windows, spread over the file."""
    import numpy as np

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    length, step = round(seconds * SAMPLE_RATE), round((seconds + .25) * SAMPLE_RATE)
    starts = [s for s in range(4000, len(audio) - length - 4000 + 1, step)
              if np.sqrt(np.mean(audio[s:s + length].astype(float) ** 2)) > 1e-3]
    return [starts[i] for i in np.linspace(0, len(starts) - 1, min(6, len(starts))).round().astype(int)]


def make_snippets(cfg):
    """Resample to mono 16 kHz and save disjoint windows; assign no trial roles yet."""
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from scipy.signal import resample_poly
    from tqdm.auto import tqdm

    audio_records, _ = read_metadata(cfg)
    severity = read_severity(cfg)
    rows, exclusions = [], []
    cfg.snippets.mkdir(parents=True, exist_ok=True)
    for record in tqdm(audio_records, desc="Snippets"):
        sid = normalize_id(re.sub(r"[_ ]*E[_ ]?NSS$", "", Path(record["filename"]).stem, flags=re.I))
        if sid not in severity:
            exclusions.append(dict(speaker=sid, reason="severity ID mismatch"))
            continue
        audio, sr = sf.read(cfg.data / "audio" / record["filename"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite samples: {record['filename']}")
        divisor = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype("float32")
        starts = snippet_starts(audio, cfg.seconds)
        if len(starts) < 4:
            exclusions.append(dict(speaker=sid, reason="fewer than 4 usable clips"))
            continue
        for start in starts:
            end = start + round(cfg.seconds * SAMPLE_RATE)
            path = cfg.snippets / f"{sid}_{start}.wav"
            sf.write(path, audio[start:end], SAMPLE_RATE, subtype="FLOAT")
            rows.append(dict(speaker=sid, severity=severity[sid], path=str(path),
                             start=start, end=end, source_hash=record["content_details"]["sha256_hash"]))
    cfg.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions, columns=["speaker", "reason"]).to_csv(cfg.out / "exclusions.csv", index=False)
    if not rows:
        raise ValueError("No usable snippets. Check audio files and metadata IDs.")
    return pd.DataFrame(rows)


def split_speakers(clips, seed=2026):
    """60/20/20 speaker-disjoint split, stratified by severity (never split snippets)."""
    from sklearn.model_selection import train_test_split

    people = clips[["speaker", "severity"]].drop_duplicates()
    if people.speaker.duplicated().any():
        raise ValueError("A speaker has more than one severity label")
    train, heldout = train_test_split(people, test_size=.4, stratify=people.severity, random_state=seed)
    dev, test = train_test_split(heldout, test_size=.5, stratify=heldout.severity, random_state=seed)
    mapping = {sid: name for name, frame in [("train", train), ("dev", dev), ("test", test)]
               for sid in frame.speaker}
    result = clips.copy()
    result["split"] = result.speaker.map(mapping)
    validate_clips(result)
    return result


def assign_roles(clips, seed=2026, n_enroll=2, n_probe=2):
    """Shuffle within each speaker, then assign disjoint enrollment/probe snippets.

    This mixes recording tasks; it does not label or match sentences and vowels.
    For a task-specific evaluation, filter to annotated clips before calling this.
    Existing speaker splits are preserved. Stable even if dataframe rows reorder.
    """
    if n_enroll < 1 or n_probe < 1:
        raise ValueError("At least one enrollment and one probe clip are required")
    result = clips.copy().reset_index(drop=True)
    result["role"] = "unused"
    for sid, group in result.groupby("speaker", sort=True):
        indices = group.sort_values("start").index.tolist()
        if len(indices) < n_enroll + n_probe:
            raise ValueError(f"Too few snippets for {sid}")
        random.Random(f"{seed}:{sid}").shuffle(indices)
        result.loc[indices[:n_enroll], "role"] = "enroll"
        result.loc[indices[n_enroll:n_enroll + n_probe], "role"] = "probe"
    return result


def validate_clips(clips):
    """Reject identity/source leakage and overlapping windows before training."""
    if clips.groupby("speaker").split.nunique().max() != 1:
        raise ValueError("Speaker leakage across splits")
    if clips.groupby("source_hash").split.nunique().max() != 1:
        raise ValueError("Source recording leakage across splits")
    for _, group in clips.groupby("source_hash"):
        ordered = group.sort_values("start")
        if (ordered.start.to_numpy()[1:] < ordered.end.to_numpy()[:-1]).any():
            raise ValueError("Overlapping or duplicated snippets in a recording")


def prepare(cfg):
    """Download data on Colab, make snippets, split speakers, shuffle trial roles."""
    download_data(cfg)
    clips = assign_roles(split_speakers(make_snippets(cfg), cfg.seed), cfg.seed)
    clips.to_csv(cfg.out / "snippets.csv", index=False)
    print(clips.drop_duplicates("speaker").groupby(["split", "severity"]).size().unstack(fill_value=0))
    return clips


def read_clips(cfg):
    import pandas as pd

    return pd.read_csv(cfg.out / "snippets.csv", dtype={"speaker": str, "source_hash": str})


def download_model(cfg):
    """Fetch pinned VoicePrivacy code and the official WavLM + ECAPA weights on Colab."""
    base = f"https://raw.githubusercontent.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026/{REVISION}"
    for relative in ["evaluation/privacy/asv/WavLM.py", "evaluation/privacy/asv/modules.py",
                     "evaluation/privacy/asv/ecapa_model_wavlm_24layer.py", "LICENSE"]:
        download_file(f"{base}/{relative}", cfg.models / "source" / relative)
    names = ["WavLM-Large.pt", "embedding_model.ckpt"]
    if not all((cfg.models / name).is_file() for name in names):
        archive = download_file("https://duke.app.box.com/shared/static/wo07rwi5xcgam1v6qlzfnzb536rqug14",
                                cfg.models / "asv_ssl.zip")
        with zipfile.ZipFile(archive) as zipped:
            for name in names:
                member = next(n for n in zipped.namelist() if Path(n).name == name)
                temporary = cfg.models / (name + ".part")
                with zipped.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(cfg.models / name)
        archive.unlink()


def load_embedding(model, path):
    """Load either official or fine-tuned ECAPA weights into an existing model."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.embedding.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)


def load_model(cfg):
    """Load cached weights; no download. Imports GPU libraries only when called."""
    import torch
    import torch.nn.functional as F
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime on Colab")
    for name in ["WavLM-Large.pt", "embedding_model.ckpt"]:
        if not (cfg.models / name).is_file():
            raise FileNotFoundError(f"Missing {name}; first call pvqd.download_model(cfg)")
    sys.path.insert(0, str(cfg.models / "source"))
    wavlm_module = importlib.import_module("evaluation.privacy.asv.WavLM")
    ecapa_module = importlib.import_module("evaluation.privacy.asv.ecapa_model_wavlm_24layer")

    class Verifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            checkpoint = torch.load(cfg.models / "WavLM-Large.pt", map_location="cpu", weights_only=False)
            self.wavlm = wavlm_module.WavLM(wavlm_module.WavLMConfig(checkpoint["cfg"]))
            self.wavlm.load_state_dict(checkpoint["model"], strict=True)
            self.wavlm.requires_grad_(False)
            self.embedding = ecapa_module.ECAPA_TDNN_test()
            self.fbank = Fbank(n_mels=80)
            self.normalizer = InputNormalization(norm_type="sentence", std_norm=False)
            load_embedding(self, cfg.models / "embedding_model.ckpt")

        def train(self, mode=True):
            super().train(mode)
            self.wavlm.eval()
            return self

        def forward(self, wavs):
            fbank = self.normalizer(self.fbank(wavs), torch.ones(len(wavs), device=wavs.device))
            ssl_wavs = F.layer_norm(wavs, (wavs.shape[-1],)) if self.wavlm.cfg.normalize else wavs
            with torch.no_grad():
                features, _ = self.wavlm.extract_features(
                    ssl_wavs, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)
                layers = torch.stack([x for x, _ in features[1][1:]]).permute(2, 0, 1, 3)
            return self.embedding(fbank, layers).squeeze(1)

    return Verifier().to(cfg.device).eval()


def waveform_batch(frame, device="cuda"):
    import numpy as np
    import soundfile as sf
    import torch

    waves = []
    for path in frame.path:
        audio, sr = sf.read(path, dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite mono 16 kHz audio: {path}")
        waves.append(audio)
    return torch.from_numpy(np.stack(waves)).to(device)


def extract_embeddings(model, frame, batch_size=4):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if frame.empty or batch_size < 1:
        raise ValueError("Need clips and a positive batch size")
    model.eval()
    device = next(model.parameters()).device
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            raw = model(waveform_batch(batch, device))
            if not torch.isfinite(raw).all() or (raw.norm(dim=-1) < 1e-12).any():
                raise ValueError(f"Non-finite/zero embedding for {batch.path.tolist()}")
            vectors.append(F.normalize(raw, dim=-1).cpu().numpy())
    return np.concatenate(vectors)


def cosine_trials(frame, vectors):
    """Build all probe/claim scores from row-aligned embeddings; useful for debugging."""
    import numpy as np
    import pandas as pd

    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(frame) or not np.isfinite(vectors).all():
        raise ValueError("Need one finite embedding per dataframe row")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Zero embedding")
    vectors = vectors / norms
    identities = frame.loc[frame.role == "enroll", "speaker"].unique()
    probe_mask = (frame.role == "probe").to_numpy()
    probe = frame.loc[probe_mask]
    if not len(identities) or probe.empty or not set(probe.speaker).issubset(set(identities)):
        raise ValueError("Every probe identity needs enrollment clips")
    prototypes = np.stack([vectors[((frame.role == "enroll") & (frame.speaker == sid)).to_numpy()].mean(0)
                           for sid in identities])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Zero enrollment prototype")
    scores = vectors[probe_mask] @ (prototypes / norms).T
    targets = probe.speaker.to_numpy()[:, None] == identities[None, :]
    return pd.DataFrame(dict(probe_path=np.repeat(probe.path.to_numpy(), len(identities)),
                             probe=np.repeat(probe.speaker.to_numpy(), len(identities)),
                             claim=np.tile(identities, len(probe)), target=targets.ravel().astype(int),
                             score=scores.ravel()))


def score_trials(model, clips, split="dev", batch_size=4):
    frame = clips[(clips.split == split) & clips.role.isin(["enroll", "probe"])].copy()
    return cosine_trials(frame, extract_embeddings(model, frame, batch_size))


def eer(trials):
    """Return EER as a fraction (0..1); reject invalid scores instead of hiding them."""
    import numpy as np
    from sklearn.metrics import roc_curve

    targets, scores = np.asarray(trials["target"]), np.asarray(trials["score"])
    if not np.isfinite(scores).all() or set(np.unique(targets)) != {0, 1}:
        raise ValueError("EER needs finite scores and both target/non-target trials")
    far, tpr, _ = roc_curve(targets, scores, drop_intermediate=False)
    return float(np.interp(0, far + tpr - 1, far))


def sample_epoch(clips, seed=2026, examples_per_speaker=6, balance="speaker"):
    """Equal examples per speaker by default; optionally equal totals per severity."""
    import numpy as np

    if balance not in {"speaker", "severity"} or examples_per_speaker < 1:
        raise ValueError("Use balance='speaker' or 'severity', with positive examples_per_speaker")
    train = clips[clips.split == "train"]
    if train.empty:
        raise ValueError("No training speakers")
    rng, indices = np.random.default_rng(seed), []
    people = train[["speaker", "severity"]].drop_duplicates()
    groups = [people] if balance == "speaker" else [g for _, g in people.groupby("severity")]
    budget = max(len(group) for group in groups) * examples_per_speaker
    for group in groups:
        speaker_order = np.concatenate([rng.permutation(group.speaker.to_numpy())
                                        for _ in range(math.ceil(budget / len(group)))])[:budget]
        for sid in speaker_order:
            indices.append(int(rng.choice(np.flatnonzero((train.speaker == sid).to_numpy()))))
    rng.shuffle(indices)
    return train.iloc[indices].reset_index(drop=True)


def train(cfg, clips, model=None):
    """Fine-tune ECAPA and a new speaker classifier; freeze WavLM and BN statistics.

    If model is omitted, start from cached pretrained weights. The best checkpoint
    uses development EER only. On tied EERs, the earlier checkpoint is retained.
    """
    import numpy as np
    import pandas as pd
    import torch
    from speechbrain.lobes.models.ECAPA_TDNN import Classifier
    from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper
    from tqdm.auto import tqdm

    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    validate_clips(clips)
    seed_everything(cfg.seed)
    model = load_model(cfg) if model is None else model
    device = next(model.parameters()).device
    labels = {sid: i for i, sid in enumerate(sorted(clips.loc[clips.split == "train", "speaker"].unique()))}
    classifier = Classifier(input_size=192, out_neurons=len(labels)).to(device)
    loss_function = LogSoftmaxWrapper(AdditiveAngularMargin(margin=.2, scale=30))
    embedding_params, head_params = list(model.embedding.parameters()), list(classifier.parameters())
    parameters = embedding_params + head_params
    optimizer = torch.optim.AdamW([{"params": embedding_params, "lr": cfg.lr},
                                  {"params": head_params, "lr": cfg.classifier_lr}], weight_decay=1e-4)
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    history, best_eer, best_epoch = [], float("inf"), None
    for epoch in range(1, cfg.epochs + 1):
        epoch_clips = sample_epoch(clips, cfg.seed + epoch, cfg.examples_per_speaker, cfg.balance)
        model.train()
        for layer in model.embedding.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()  # Keep pretrained running statistics; affine parameters still train.
        classifier.train()
        losses, correct, count = [], 0, 0
        for start in tqdm(range(0, len(epoch_clips), cfg.batch_size), desc=f"Epoch {epoch}/{cfg.epochs}"):
            batch = epoch_clips.iloc[start:start + cfg.batch_size]
            targets = torch.tensor([labels[sid] for sid in batch.speaker], device=device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(model(waveform_batch(batch, device)).unsqueeze(1))
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; stop and reload clean pretrained weights")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            correct += (logits.detach().argmax(-1) == targets).sum().item()
            count += len(batch)
        dev_eer = eer(score_trials(model, clips, "dev", cfg.batch_size))
        row = dict(epoch=epoch, loss=float(np.mean(losses)), train_accuracy=correct / count, dev_eer=dev_eer)
        history.append(row)
        print(row)
        if dev_eer < best_eer:
            best_eer, best_epoch = dev_eer, epoch
            torch.save(model.embedding.state_dict(), cfg.out / "best_finetuned.pt")
        pd.DataFrame(history).to_csv(cfg.out / "training.csv", index=False)
    print(f"Selected epoch {best_epoch}: development EER {100 * best_eer:.2f}%")
    load_embedding(model, cfg.out / "best_finetuned.pt")
    return model, pd.DataFrame(history)


def evaluate(cfg, clips, model=None, split="test", checkpoints=None):
    """Compare checkpoints on the same explicit trials; save scores and EER table."""
    import pandas as pd

    model = load_model(cfg) if model is None else model
    if checkpoints is None:
        checkpoints = {"pretrained": cfg.models / "embedding_model.ckpt",
                       "finetuned": cfg.out / "best_finetuned.pt"}
    cfg.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in checkpoints.items():
        load_embedding(model, path)
        trials = score_trials(model, clips, split, cfg.batch_size)
        trials.to_csv(cfg.out / f"{name}_{split}_scores.csv", index=False)
        rows.append(dict(model=name, EER_percent=100 * eer(trials),
                         target_pairs=int(trials.target.sum()), non_target_pairs=int((1 - trials.target).sum())))
    results = pd.DataFrame(rows)
    results.to_csv(cfg.out / f"{split}_eer.csv", index=False)
    print(results.to_string(index=False))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "download-model", "train", "evaluate", "run"])
    defaults = Config()
    for field in ["root", "metadata_dir", "seed", "seconds", "batch_size", "epochs", "lr", "classifier_lr",
                  "examples_per_speaker", "device"]:
        value = getattr(defaults, field)
        parser.add_argument("--" + field.replace("_", "-"), default=value, type=type(value))
    parser.add_argument("--balance", choices=["speaker", "severity"], default=defaults.balance)
    args = vars(parser.parse_args(argv))
    step = args.pop("step")
    cfg = Config(**args)
    if step in {"prepare", "run"}:
        clips = prepare(cfg)
    if step in {"download-model", "run"}:
        download_model(cfg)
    if step in {"train", "evaluate"}:
        clips = read_clips(cfg)
    if step in {"train", "run"}:
        model, _ = train(cfg, clips)
    if step == "evaluate":
        evaluate(cfg, clips)
    elif step == "run":
        evaluate(cfg, clips, model)


if __name__ == "__main__":
    main()
