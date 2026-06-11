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

Try:

```powershell
python -m sansoku_ai.tests.run_tests
python -m sansoku_ai.scripts.benchmark_endgame --remaining 6 --samples 3
python -m sansoku_ai.scripts.play_match --first ab2 --second greedy --games 2 --endgame 6
python -m sansoku_ai.scripts.generate_mixed_games --games 10 --endgame 6 --move-limit 12
python -m sansoku_ai.scripts.play_match --first ru3 --second ab3 --games 2 --endgame 4 --move-limit 8 --ranker-model models/linear_ranker_best.json
python -m sansoku_ai.scripts.run_iteration --name iter001 --games 1000 --workers 8 --ranker-model models/linear_ranker_v2.json
```
