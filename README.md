# Sansoku AI Sandbox

This is a small Python sandbox for building AI for the 6x6 number-placement
game "Sansoku".

Current slice:

- Exact rule engine and legal move generation.
- Safe dominance removal: for the same cell and same ones digit, keep only the
  largest value.
- Negamax alpha-beta search.
- Endgame exact search when `remaining <= 8`.
- A heuristic evaluation that keeps first/second asymmetry and empty-center-empty
  trap features as evaluation only, not as pruning.
- Greedy and alpha-beta match runner.
- Ranker-union root search and an automated self-play/reanalysis iteration loop.
- Parallel/resumable mixed-game generation for RunPod or other multi-core hosts.
- A first PyTorch NN ranker scaffold: board CNN + legal-move-list policy scores.
- NN dataset building uses the four safe Sansoku board symmetries by default in
  the iteration loop, with train/validation split done before augmentation.
- Arena and match results use komi 16 by default; pass `--komi 0` for raw score
  margins.

Try:

```powershell
python -m sansoku_ai.tests.run_tests
python -m sansoku_ai.scripts.benchmark_endgame --remaining 6 --samples 3
python -m sansoku_ai.scripts.play_match --first ab2 --second greedy --games 2 --endgame 6
python -m sansoku_ai.scripts.generate_mixed_games --games 10 --endgame 6 --move-limit 12
python -m sansoku_ai.scripts.play_match --first ru3 --second ab3 --games 2 --endgame 4 --move-limit 8 --ranker-model models/linear_ranker_best.json
python -m sansoku_ai.scripts.run_iteration --name iter001 --games 1000 --workers 8 --ranker-model models/linear_ranker_v2.json
```

NN ranker smoke path after installing PyTorch:

```powershell
python -m sansoku_ai.scripts.train_nn_ranker --train data/train_v2.jsonl --val data/val_v2.jsonl --epochs 5 --output models/nn_ranker_v1.pt
python -m sansoku_ai.scripts.evaluate_ranker models/nn_ranker_v1.pt data/val_v2.jsonl
python -m sansoku_ai.scripts.compare_rankers models/linear_ranker_v2.json models/nn_ranker_v1.pt data/val_v2.jsonl --name-a linear --name-b nn
python -m sansoku_ai.scripts.arena --candidate ru3 --opponent-mix ab2:0.5,ab3:0.5 --games 20 --ranker-model models/nn_ranker_v1.pt --candidate-move-limit 8 --opponent-move-limit 8
```

Policy/value bootstrap after installing PyTorch:

```powershell
python -m sansoku_ai.scripts.train_policy_value --train data/train_v2.jsonl --val data/val_v2.jsonl --epochs 20 --batch-size 128 --policy-target-mode policy --value-target-mode search --output models/policy_value_v1.pt
python -m sansoku_ai.scripts.evaluate_policy_value models/policy_value_v1.pt data/val_v2.jsonl
python -m sansoku_ai.scripts.arena --candidate pvab3 --opponent-mix ab2:0.5,ab3:0.5 --games 20 --policy-value-model models/policy_value_v1.pt --candidate-move-limit 8 --opponent-move-limit 8 --nn-value-weight 0.25 --komi 16
python -m sansoku_ai.scripts.arena --candidate puct100 --opponent-mix ab2:0.5,ab3:0.5 --games 10 --policy-value-model models/policy_value_v1.pt --puct-simulations 100 --puct-batch-size 8 --puct-leaf-depth 1 --puct-leaf-weight 0.25 --puct-leaf-move-limit 8 --komi 16
python -m sansoku_ai.scripts.generate_pv_selfplay --model models/policy_value_v1.pt --output data/pv_selfplay_100.jsonl --games 100 --player puct100 --puct-simulations 100 --puct-batch-size 8 --puct-leaf-depth 1 --puct-leaf-weight 0.25 --puct-leaf-move-limit 8 --resume
python -m sansoku_ai.scripts.build_pv_training_dataset data/pv_selfplay_100.jsonl --output data/pv_dataset.jsonl --train-output data/pv_train.jsonl --val-output data/pv_val.jsonl --symmetry-augment --komi 16 --target-komi 0
python -m sansoku_ai.scripts.train_policy_value --train data/pv_train.jsonl --val data/pv_val.jsonl --epochs 20 --batch-size 128 --policy-target-mode policy --value-target-mode search --output models/policy_value_from_puct.pt
```

`pvabN` uses the policy/value net for move ordering and blends the value head
into alpha-beta leaf evaluation. Start with a small `--nn-value-weight` such as
`0.25`; use `1.0` only after the value head proves reliable. `puctN` writes
root visit distributions into `mcts_policy` when generating policy/value
self-play games. It also supports `--puct-batch-size`, so multiple leaf
evaluations can be batched into one NN call. Early PUCT should usually use
`--puct-leaf-depth 1 --puct-leaf-weight 0.25`, so the young value head is
corrected by shallow alpha-beta. Self-play uses root Dirichlet noise and
visit-count sampling by default; evaluation games do not. The value target
remains raw/no-komi by default (`--target-komi 0`) because `pvab` and `puct` add
komi correction during search.

Policy/value repeat-and-promote cycle:

```powershell
python -m sansoku_ai.scripts.run_pv_cycle --prefix pv_runpod --cycles 3 --initial-model models/policy_value_v1.pt --games 1000 --puct-simulations 100 --puct-batch-size 8 --puct-leaf-depth 1 --puct-leaf-weight 0.25 --puct-leaf-move-limit 8 --train-epochs 20 --promote-games 40 --reanalyze-workers 4 --continue-on-fail
```

The cycle generates PUCT self-play, trains from MCTS visit policies, evaluates
against fixed alpha-beta opponents, and promotes the new model only if it beats
the previous policy/value champion with balanced first/second games. Promotion
losses are mined automatically: candidate losses, second-player losses, and
positions that allowed opponent high-value moves are reanalyzed with alpha-beta
and fed into later policy/value cycles as extra teacher data.

Automated NN self-play/reanalysis/training iteration:

```powershell
python -m sansoku_ai.scripts.run_nn_iteration --name nn_iter001 --games 5000 --workers 8 --ranker-model models/nn_ranker_v2.pt
```

Repeat-and-promote cycle for RunPod:

```powershell
python -m sansoku_ai.scripts.run_nn_cycle --prefix nn_runpod --cycles 3 --games 5000 --workers 8 --initial-ranker-model models/nn_ranker_v2.pt
```
