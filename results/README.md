# Experiment result archives

Each completed experiment has one directory:

```text
results/<code-revision>/<run-id>/
├── README.md                   # concise result and provenance summary (regular Git)
├── SHA256SUMS                  # checksums of every file inside the archive (regular Git)
├── artifacts.tar.zst          # complete result payload (Git LFS)
└── artifacts.tar.zst.sha256   # checksum of the compressed archive (regular Git)
```

Only `artifacts.tar.zst` belongs in Git LFS. Keeping summaries and checksum files
in regular Git makes results reviewable without downloading hundreds of megabytes,
while one archive avoids a separate LFS pointer and request for every checkpoint.

## What to archive

Include the complete result payload needed to audit or continue the experiment:

- resolved configurations and environment records;
- metrics, validation outputs, and scheduler/console logs;
- every retained checkpoint, including initialization and the final checkpoint;
- preflight and smoke-test evidence that belongs to the run.

Exclude datasets, virtual environments, package/model caches, temporary files, and
incomplete `*.tmp` files. Never remove the source artifacts until the archive has
been uploaded, checked from the remote, and accepted under the applicable retention
policy.

## Create and verify an archive

The source must be a finalized, immutable run directory. The destination must not
already exist. `zstd` and GNU `tar` must be available.

```bash
scripts/archive-results.sh \
  /absolute/path/to/finalized-run \
  results/<code-revision>/<run-id>
```

The script creates a deterministic archive: paths are sorted, ownership is
normalized to root, permissions are normalized, timestamps are fixed, compression
uses one thread, and archive extraction is verified against `SHA256SUMS`. Review and
write the run-specific `README.md` separately; it is intentionally not generated
from the raw metrics.

Track archives, but not summaries or manifests, with Git LFS:

```gitattributes
results/**/*.tar.zst filter=lfs diff=lfs merge=lfs -text
```

Before pushing, require all of these checks:

```bash
git check-attr filter -- results/<code-revision>/<run-id>/artifacts.tar.zst
git lfs ls-files
git lfs fsck
```

After cloning or pulling, verify and extract an archive with:

```bash
cd results/<code-revision>/<run-id>
git lfs pull --include='results/<code-revision>/<run-id>/artifacts.tar.zst'
sha256sum --check artifacts.tar.zst.sha256
mkdir extracted
tar --zstd --extract --file artifacts.tar.zst --directory extracted
(cd extracted && sha256sum --check ../SHA256SUMS)
```

GitHub may retain orphaned LFS objects for quota accounting after history rewrites.
Avoid uploading a provisional layout. If an archive approaches the hosting
provider's per-object limit, split it into documented logical payloads rather than
silently omitting artifacts.
