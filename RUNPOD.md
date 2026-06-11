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

If interrupted, rerun the same command. Reanalysis stages use `--resume`
internally when their output already exists.

Outputs:

```text
data/iterations/iter001/
models/linear_ranker_iter001.json
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
