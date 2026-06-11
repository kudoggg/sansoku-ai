# Setup

## Minimum local setup

Install these first:

1. Python 3.12 or 3.13 for Windows.
2. Git for Windows.
3. Visual Studio Code.

The current alpha-beta engine does not require PyTorch yet.

After installing Python, open PowerShell in this folder and run:

```powershell
python -m sansoku_ai.tests.run_tests
python -m sansoku_ai.scripts.benchmark_endgame --remaining 6 --samples 3
python -m sansoku_ai.scripts.play_match --first ab2 --second greedy --games 2 --endgame 6 --move-limit 12
```

If `python` is not found, restart PowerShell. If it is still not found, reinstall
Python and enable the PATH option in the installer.

## Later neural-network setup

Install PyTorch only when policy/value training begins.

Use the official PyTorch selector and choose:

- OS: Windows for local work, Linux for RunPod.
- Package: Pip.
- Language: Python.
- Compute Platform: CUDA if using an NVIDIA GPU.

RunPod images often already include CUDA/PyTorch. For RunPod, prefer starting
from a PyTorch template before manually installing CUDA.

## Mixed game generation

For diverse but still reasonable positions, generate games with a random-ish
four-ply opening followed by a random mix of `ab2` and `ab3`.

Fast local smoke test:

```powershell
python -m sansoku_ai.scripts.generate_mixed_games --games 10 --endgame 4 --move-limit 10 --ab2-prob 0.5
```

Higher-quality but slower:

```powershell
python -m sansoku_ai.scripts.generate_mixed_games --games 10 --endgame 6 --move-limit 10 --ab2-prob 0.8
```

Write JSONL records:

```powershell
python -m sansoku_ai.scripts.generate_mixed_games --games 100 --endgame 4 --move-limit 10 --output data/mixed_100.jsonl
```

Generate stronger mixed games using ranker-union players:

```powershell
python -m sansoku_ai.scripts.generate_mixed_games --games 100 --endgame 4 --move-limit 8 --policy-mix ab2:0.3,ab3:0.4,ru2:0.15,ru3:0.15 --ranker-model models/linear_ranker_best.json --output data/mixed_ru_100.jsonl
```

Parallel and resumable mixed game generation:

```powershell
python -m sansoku_ai.scripts.generate_mixed_games --games 1000 --workers 4 --endgame 4 --move-limit 8 --policy-mix ab2:0.3,ab3:0.4,ru2:0.15,ru3:0.15 --ranker-model models/linear_ranker_v2.json --output data/mixed_ru_1000.jsonl
python -m sansoku_ai.scripts.generate_mixed_games --games 1000 --workers 4 --endgame 4 --move-limit 8 --policy-mix ab2:0.3,ab3:0.4,ru2:0.15,ru3:0.15 --ranker-model models/linear_ranker_v2.json --output data/mixed_ru_1000.jsonl --resume
```

Analyze written values:

```powershell
python -m sansoku_ai.scripts.analyze_games data/mixed_100.jsonl
```

Sample positions and reanalyze them:

```powershell
python -m sansoku_ai.scripts.sample_positions data/mixed_100.jsonl --output data/positions_100.jsonl
python -m sansoku_ai.scripts.reanalyze_positions data/positions_100.jsonl --output data/reanalyzed_100_d4.jsonl --depth 4 --endgame 6 --root-limit 10 --move-limit 10
```

Parallel and resumable reanalysis:

```powershell
python -m sansoku_ai.scripts.reanalyze_positions data/positions_1000.jsonl --output data/reanalyzed_parallel.jsonl --depth 5 --endgame 4 --root-limit 16 --move-limit 12 --workers 4 --progress-every 50
python -m sansoku_ai.scripts.reanalyze_positions data/positions_1000.jsonl --output data/reanalyzed_parallel.jsonl --depth 5 --endgame 4 --root-limit 16 --move-limit 12 --workers 4 --resume --progress-every 50
```

Select hard positions for heavier reanalysis:

```powershell
python -m sansoku_ai.scripts.select_hard_positions data/reanalyzed_all_d3_fast.jsonl --output data/hard_500_from_d3.jsonl --limit 500
```

Compare light and heavier reanalysis:

```powershell
python -m sansoku_ai.scripts.compare_reanalysis data/hard_500_from_d3.jsonl data/hard_500_d4.jsonl --d5-output data/hard_100_for_d5.jsonl --d5-limit 100
```

Build a weighted training dataset from available reanalysis files:

```powershell
python -m sansoku_ai.scripts.build_training_dataset --output data/training_dataset.jsonl --train-output data/train.jsonl --val-output data/val.jsonl
```

Train and evaluate a dependency-free linear ranker:

```powershell
python -m sansoku_ai.scripts.train_linear_ranker --epochs 20 --output models/linear_ranker.json
python -m sansoku_ai.scripts.evaluate_ranker models/linear_ranker.json data/val.jsonl
```

Use the ranker only as root-candidate union support:

```powershell
python -m sansoku_ai.scripts.play_match --first ru3 --second ab3 --games 2 --endgame 4 --move-limit 8 --ranker-model models/linear_ranker_best.json
```

Compare two rankers on the same validation set:

```powershell
python -m sansoku_ai.scripts.compare_rankers models/linear_ranker_best.json models/linear_ranker_v2.json data/val_v2.jsonl --name-a v1 --name-b v2
```

Run one automated local iteration:

```powershell
python -m sansoku_ai.scripts.run_iteration --name iter001 --games 1000 --workers 4 --ranker-model models/linear_ranker_v2.json
python -m sansoku_ai.scripts.run_iteration --name iter001_eval_more --games 1000 --workers 4 --ranker-model models/linear_ranker_v2.json --arena-strong-games 10 --arena-full-games 6 --arena-strong-full-games 4
```

Run an arena only:

```powershell
python -m sansoku_ai.scripts.arena --candidate ru3 --opponent-mix ab2:0.5,ab3:0.5 --games 20 --ranker-model models/linear_ranker_v2.json --candidate-move-limit 8 --opponent-move-limit 8
python -m sansoku_ai.scripts.arena --candidate ru3 --opponent-mix ab3:0.5,ab4:0.5 --games 10 --ranker-model models/linear_ranker_v2.json --candidate-move-limit 8 --opponent-move-limit 8
python -m sansoku_ai.scripts.arena --candidate ru3 --opponent-mix ab2:0.5,ab3:0.5 --games 10 --ranker-model models/linear_ranker_v2.json --full-candidate --full-opponent
python -m sansoku_ai.scripts.arena --candidate ru3 --opponent-mix ab3:0.5,ab4:0.5 --games 6 --ranker-model models/linear_ranker_v2.json --full-candidate --full-opponent
```

Summarize and package an iteration:

```powershell
python -m sansoku_ai.scripts.summarize_iteration iter001
python -m sansoku_ai.scripts.package_iteration iter001
```

## Development rule

Keep exact rules and hypotheses separate:

- Exact: legal move generation, dominance, alpha-beta, transposition table,
  exact endgame.
- Hypothesis: corner maximum, zero/one pruning, triplet pruning, trap bias.

Hypotheses should start as move ordering or evaluation features, not hard
pruning.
