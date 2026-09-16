#!/usr/bin/env python3
# Fine-grained worker: runs its static slice of the full (dim,seed,corpus) run-list.
# Idempotent: skips any run whose per-run result already exists in S3. Reuses
# titans_lr_horizon.run() (hardened overfit checks). One run per GPU at a time.
import json, subprocess, types, os, sys, torch
import titans_lr_horizon as T

S3 = "s3://<your-bucket>/runs"
WID = int(os.environ.get("WORKER_ID", sys.argv[1] if len(sys.argv) > 1 else 0))
NUM = int(os.environ.get("NUM_WORKERS", "11"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T4 = 20_000_000
CORPORA = ["AR_phi0.00", "AR_phi0.90", "electricity_15min", "StarLightCurves"]

RUNS = []
for seed in [0, 1, 2, 3, 4]:                       # dim 384: N=5
    for c in CORPORA: RUNS.append({"dim": 384, "seed": seed, "corpus": c})
for dim in [64, 512]:                               # width sweep: N=3
    for seed in [0, 1, 2]:
        for c in CORPORA: RUNS.append({"dim": dim, "seed": seed, "corpus": c})
for seed in [0, 1, 2]:                              # 2nd real corpus (dim 384)
    RUNS.append({"dim": 384, "seed": seed, "corpus": "UWaveGestureLibraryAll"})

def loader(corpus, seed):
    if corpus == "AR_phi0.00": return T.ar1_big(T4, 0.0, seed)
    if corpus == "AR_phi0.90": return T.ar1_big(T4, 0.9, seed)
    if corpus == "electricity_15min": return T.stream_chronos("electricity_15min", T4)
    return T.load_ucr("/opt/data", corpus)          # StarLightCurves / UWaveGestureLibraryAll

def exists(key):
    return subprocess.run(["aws", "s3", "ls", key, "--region", os.environ.get("AWS_REGION", "us-east-1")],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

print(f"worker {WID}/{NUM} on {DEV}; {sum(1 for i in range(len(RUNS)) if i % NUM == WID)} runs assigned", flush=True)
for i, r in enumerate(RUNS):
    if i % NUM != WID: continue
    key = f"{S3}/{r['corpus']}_dim{r['dim']}_seed{r['seed']}.json"
    if exists(key): print(f"skip (done) {key}", flush=True); continue
    try:
        tr, va = T.quantize(loader(r["corpus"], r["seed"]))
    except Exception as e:
        print(f"LOAD_FAIL {r} {str(e)[:150]}", flush=True); continue
    a = types.SimpleNamespace(dim=r["dim"], seed=r["seed"], steps=6000, seq_len=256,
                              batch=8, lr=2e-4, seg=8, passage=1024)
    row = T.run(r["corpus"], tr, va, a, DEV); row["dim"] = r["dim"]; row["seed"] = r["seed"]
    open("/tmp/one.json", "w").write(json.dumps(row))
    subprocess.run(["aws", "s3", "cp", "/tmp/one.json", key, "--region", os.environ.get("AWS_REGION", "us-east-1")], check=False)
    del tr, va
    if DEV == "cuda": torch.cuda.empty_cache()
print(f"worker {WID} done", flush=True)
