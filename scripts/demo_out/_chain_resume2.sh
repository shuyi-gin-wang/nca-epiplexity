#!/usr/bin/env bash
# Wait for the in-flight 100-gen resume (PID 410) to finish, then chain a second
# resume for 100 more generations (effectively gens 100->200 of this lineage).
set -u
cd /c/Users/gin.wang/nca

echo "[chain] waiting for PID 410 to finish ($(date))"
while ps -W 2>/dev/null | grep -qE '^[[:space:]]*410[[:space:]]'; do
  sleep 60
done
echo "[chain] PID 410 gone ($(date))"

SEED1=scripts/demo_out/sweep_transformer_direct_K16_pdrop0_resume/best_ever_params.pt
if [ ! -f "$SEED1" ]; then
  echo "[chain] WARNING: $SEED1 missing (run 1 may have crashed); falling back to original best."
  SEED1=scripts/demo_out/sweep_transformer_direct_K16_pdrop0/best_ever_params.pt
fi
echo "[chain] launching resume2 from $SEED1 ($(date))"

.venv/Scripts/python.exe scripts/epiplexity/trainers/evolve_gzip.py \
  --run-name sweep_transformer_direct_K16_pdrop0_resume2 \
  --resume-from "$SEED1" \
  --p-drop 0.0 \
  --gzip-mode threshold --gzip-threshold 0.3 \
  --probe-arch transformer --probe-hidden 0 \
  --horizon-mode direct --probe-horizon 16 \
  --pop-size 128 --n-generations 100 \
  --checkpoint-every 25 \
  --seed 0
echo "[chain] resume2 finished ($(date))"
