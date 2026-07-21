# postflop-solver-batch

Batch precompute farm for the flop-strategy cache of a **private poker study
tool** (a GTO study overlay used in home games among consenting friends).
Wraps the open-source [postflop-solver](https://github.com/b-inary/postflop-solver)
CFR engine.

Each cache entry covers one *(preflop range pair, SPR, canonical flop)*:
the full flop→river tree is CFR-solved to ~2%-of-pot exploitability, and the
entry stores per-combo strategy frequencies + EVs for every flop-street
decision node (~110–400KB JSON). `cache-config.json` lists the range pairs
and SPR rungs; flops are solved in real-deal frequency order.

## Usage

```sh
# locally
cargo run --release -- precompute cache-config.json [--line N] [--chunk I/K]

# fan out across GitHub Actions runners (matrix in the workflow):
gh workflow run precompute
```

Each CI job solves one interleaved chunk of one config line and uploads its
JSON entries as an artifact. Work is content-addressed and resumable — re-runs
skip already-solved flops, and merging artifacts is a copy-if-absent.

## License

AGPL-3.0, matching the upstream solver.
