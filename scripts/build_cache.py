"""Rebuild the decoded-array cache from the COMMITTED manifest. Run from the root:

    python -m scripts.build_cache                 # all 670 samples, ~5 min
    python -m scripts.build_cache --limit 10      # quick check

Why this script exists instead of rerunning build_manifest(): the committed
.project_state/manifest.json holds the frozen train/val split that every
reported number is tied to. A fresh build_manifest() drops the excluded outlier
BEFORE stratifying and reassigns 7 samples to the other split. The cache is
split-agnostic, so rebuilding only the cache reproduces the pipeline exactly.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_sample, write_cache_arrays  # noqa: E402
from src.dataset_manifest import load_manifest  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build data/cache from the frozen manifest.")
    p.add_argument("--train-dir", default="data/stage1_train")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--state-dir", default=".project_state")
    p.add_argument("--limit", type=int, default=0, help="first N samples only (0 = all)")
    a = p.parse_args(argv)

    manifest, _ = load_manifest(ROOT / a.state_dir)
    recs = manifest["samples"][: a.limit or None]  # slice end None = no limit
    cache = ROOT / a.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, r in enumerate(recs, 1):
        sid = r["sample_id"]
        image, label = load_sample(ROOT / a.train_dir / sid)
        # Modality comes from the manifest (mask-derived, authoritative), not
        # re-estimated: it decides the brightfield polarity flip baked into _img.
        write_cache_arrays(cache, sid, image, label, r["modality"])
        if i % 100 == 0:
            print(f"  {i}/{len(recs)}  {time.perf_counter() - t0:.0f}s")
    print(f"Processed {len(recs)} samples -> {cache} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
