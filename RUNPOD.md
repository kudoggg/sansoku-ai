# RunPod Notes

The current Sansoku loop is mostly CPU-bound because alpha-beta reanalysis spends
time in legal move generation and tree search. A GPU becomes important later for
policy/value/margin neural networks and NN-MCTS.

## Recommended first RunPod target

For the current pipeline:

- Prefer more vCPUs/RAM over the fastest GPU.
- RTX 3090 pods are a good first target because they have more CPU/RAM than many
  consumer GPU pods.
- RTX A5000 is cheaper and fine for small runs.
- RTX 4090/5090 become more attractive once PyTorch NN training starts.

## One automated iteration

After copying or cloning this project onto the pod:

```bash
python -m sansoku_ai.scripts.run_iteration \
  --name iter001 \
  --games 1000 \
  --workers 8 \
  --ranker-model models/linear_ranker_v2.json
```

For long runs, start it with `nohup` so browser/Web Terminal disconnects do not
stop the job:

```bash
mkdir -p logs
nohup python -m sansoku_ai.scripts.run_iteration \
  --name iter001 \
  --games 1000 \
  --workers 8 \
  --ranker-model models/linear_ranker_v2.json \
  --arena-games 40 \
  --arena-strong-games 20 \
  --arena-full-games 4 \
  --arena-strong-full-games 2 \
  > logs/iter001.log 2>&1 &
tail -f logs/iter001.log
```

If interrupted, rerun the same command. Mixed-game generation and reanalysis
stages resume from existing JSONL outputs when possible.

Outputs:

```text
data/iterations/iter001/
models/linear_ranker_iter001.json
```

The final summary prints per-step elapsed time plus arena results split by the
candidate playing first or second. Arena results use komi 16 by default; add
`--komi 0` only when you intentionally want raw score margins.

## Save iteration artifacts

After an iteration completes, package the small important outputs before
stopping or deleting anything:

```bash
python -m sansoku_ai.scripts.summarize_iteration iter001
python -m sansoku_ai.scripts.package_iteration iter001
ls -lh outputs/iter001_artifacts.tar.gz
```

This package includes the ranker model, log file, and arena JSON results. Add
`--include-data` only when you intentionally want the larger JSONL data too.

## First NN ranker run

Once PyTorch is available, train the supervised board-CNN ranker from an existing
training split:

```bash
python -m sansoku_ai.scripts.train_nn_ranker \
  --train data/train_v2.jsonl \
  --val data/val_v2.jsonl \
  --epochs 20 \
  --batch-size 128 \
  --output models/nn_ranker_v1.pt

python -m sansoku_ai.scripts.compare_rankers \
  models/linear_ranker_v2.json \
  models/nn_ranker_v1.pt \
  data/val_v2.jsonl \
  --name-a linear \
  --name-b nn
```

If the NN is competitive, use it in the same ranker-union player:

```bash
python -m sansoku_ai.scripts.arena \
  --candidate ru3 \
  --opponent-mix ab2:0.5,ab3:0.5 \
  --games 40 \
  --ranker-model models/nn_ranker_v1.pt \
  --candidate-move-limit 8 \
  --opponent-move-limit 8
```

For an unattended NN iteration:

```bash
mkdir -p logs
nohup python -m sansoku_ai.scripts.run_nn_iteration \
  --name nn_iter001 \
  --games 5000 \
  --workers 8 \
  --ranker-model models/nn_ranker_v2.pt \
  --arena-games 40 \
  --arena-strong-games 40 \
  --arena-full-games 4 \
  --arena-strong-full-games 2 \
  > logs/nn_iter001.log 2>&1 &
tail -f logs/nn_iter001.log
```

If the browser disconnects, reconnect and inspect progress:

```bash
cd /workspace/sansoku-ai
tail -n 120 logs/nn_iter001.log
ps -ef | grep run_nn_iteration | grep -v grep
```

If the job stopped partway through, rerun the same `nohup` command. Existing
complete outputs are skipped and JSONL stages resume where possible.

## Unattended NN promotion cycle

To leave RunPod running for several hours, use the cycle runner. It repeats the
NN iteration, then compares the new model against the previous champion with
komi 16. If the new model wins more games than it loses and has positive average
margin, it is promoted and the next cycle uses it. If it fails the promotion
gate, the cycle stops unless `--continue-on-fail` is added.

All generated games and arena games use root endgame exact reading when
`remaining <= --endgame`. The recommended cycle command keeps `--endgame 4`,
so even greedy/ranker-union choices switch to full alpha-beta reading for the
last four moves. Arena game counts and promotion game counts must be even so the
candidate plays first and second equally.

This is still a fixed-teacher bootstrap: positions come from the current
champion's self-play/mixed play, but labels are produced by the alpha-beta
reanalyzer. That keeps the target stable while the training distribution becomes
stronger.

NN iterations use the four safe Sansoku symmetries by default: identity, 180
degree rotation, main diagonal reflection, and anti-diagonal reflection. The
train/validation split is made before augmentation, so symmetric copies of the
same position do not leak across the split. Add `--no-symmetry-augment` only for
diagnostic runs where you want the old smaller dataset.

Make sure the initial NN model exists on the pod first, for example
`models/nn_ranker_v2.pt`.

```bash
mkdir -p logs
nohup python -m sansoku_ai.scripts.run_nn_cycle \
  --prefix nn_runpod \
  --cycles 3 \
  --initial-ranker-model models/nn_ranker_v2.pt \
  --games 5000 \
  --workers 8 \
  --hard-limit 1000 \
  --train-epochs 20 \
  --batch-size 128 \
  --arena-games 40 \
  --arena-strong-games 40 \
  --arena-full-games 4 \
  --arena-strong-full-games 2 \
  --promote-games 60 \
  > logs/nn_runpod_cycle.log 2>&1 &
tail -f logs/nn_runpod_cycle.log
```

To check whether it is still running:

```bash
ps -ef | grep run_nn_cycle | grep -v grep
tail -n 160 logs/nn_runpod_cycle.log
```

Outputs:

```text
models/nn_ranker_nn_runpod_champion.pt
models/nn_ranker_nn_runpod_001.pt
models/nn_ranker_nn_runpod_002.pt
data/iterations/nn_runpod_cycle_summary.json
outputs/nn_runpod_001_artifacts.tar.gz
outputs/nn_runpod_002_artifacts.tar.gz
```

## When RunPod is worth it

Use RunPod when:

- d5/d6 reanalysis is running for many hours locally,
- you want several iterations overnight,
- local PC heat/noise is annoying,
- PyTorch policy/value/margin training begins,
- NN-MCTS self-play begins.

Stay local when:

- testing scripts,
- generating a few hundred games,
- running d3/d4 smoke tests,
- training the tiny linear ranker.

## Current cost intuition

As of the current RunPod pricing page, example Pods prices include:

- RTX A5000: `$0.27/hr`
- RTX 3090: `$0.46/hr`
- RTX 4090: `$0.69/hr`
- RTX 5090: `$0.99/hr`
- L40S: `$0.86/hr`

Exact availability and prices can change, so check the pricing page before
starting a long run.
