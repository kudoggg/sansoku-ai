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

Automated NN self-play/reanalysis/training iteration:

```powershell
python -m sansoku_ai.scripts.run_nn_iteration --name nn_iter001 --games 5000 --workers 8 --ranker-model models/nn_ranker_v2.pt
```

Repeat-and-promote cycle for RunPod:

```powershell
python -m sansoku_ai.scripts.run_nn_cycle --prefix nn_runpod --cycles 3 --games 5000 --workers 8 --initial-ranker-model models/nn_ranker_v2.pt
```
