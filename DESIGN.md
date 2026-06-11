# Sansoku AI Strategy

## Separation of Rules and Hypotheses

Keep these separate throughout development.

Safe rules:

- Full legal move generation from adjacent occupied pairs.
- Dominance normalization: same cell and same ones digit keeps only the largest
  value.
- Alpha-beta cutoff.
- Transposition table.
- Move ordering.
- Exact endgame search.

Hypotheses:

- Corner maximum only.
- Zero/one tier pruning.
- Triplet-history pruning.
- Root candidate caps.
- Empty-center-empty traps favoring the second player.

Hypotheses should begin as evaluation features or ordering features. Promote them
to pruning only after they survive exact-search diagnostics.

## Current Baseline

The current baseline is:

- `GreedyPlayer`: chooses the highest immediate value.
- `AlphaBetaPlayer`: negamax alpha-beta with a heuristic leaf evaluator.
- `AlphaBetaSearch(endgame_exact_remaining=8)`: switches to full terminal search
  when there are at most eight moves remaining.

The heuristic evaluator includes:

- current side margin,
- max legal value,
- top-3 legal value average,
- high zero/one moves,
- edge/corner availability,
- high move count,
- empty-center-empty trap bias for the second player.

The trap feature is deliberately not pruning. It encodes the idea that shapes
like empty-9-empty can favor the second player if they survive late.

## RunPod-Oriented Roadmap

1. Make the Python engine correct and inspectable.
2. Add faster exact endgame search and profiling.
3. Generate expert data from alpha-beta plus exact endgames.
4. Train a small policy/value/margin network.
5. Add NN-guided MCTS.
6. Run self-play workers on RunPod.
7. Reanalyze hard positions with exact/alpha-beta search.
8. Promote new models only through arena matches.

## KataGo-Inspired Targets

Final neural model heads should likely include:

- policy over legal `(row, col, value)` moves,
- win value,
- margin value,
- ownership-like prediction for which player will eventually take each empty
  cell,
- expected score contribution per cell.

This matters because Sansoku is a point game with first/second asymmetry. A
single current-player win value is probably too weak.
