# Smart Water Leakage Management

Hybrid Graph Neural Network and Operations Research demo for leakage detection,
valve isolation, and degraded-network routing on the LeakDB benchmark.

## What is included

- LeakDB `Hanoi_CMH` benchmark staged in `data/raw/LeakDB/`
- Hydraulic network parsing from `.inp` files
- Temporal pressure and leak-series loading
- STGCN-style pressure anomaly detection
- Pipe closure and hydraulic re-simulation
- Ford-Fulkerson / Edmonds-Karp max-flow rerouting
- Prim MST degraded-operation planning
- Lightweight citizen-report-to-isolation agent

## Project layout

- `main.py` - end-to-end orchestrator
- `simulation/epanet_engine.py` - network parsing and hydraulic surrogate
- `ai_layer/dataset_loader.py` - LeakDB topology and temporal loader
- `ai_layer/stgcn_model.py` - STGCN-style PyTorch model
- `core_ro/ford_fulkerson.py` - max-flow rerouting
- `core_ro/prim_mst.py` - MST planning
- `agents/llm_valve_agent.py` - report parsing and isolation suggestion

## Installation

Create a virtual environment if you want isolation, then install:

```powershell
pip install -r requirements.txt
```

Notes:

- The project runs with the core dependencies `numpy`, `pandas`, `networkx`, and `torch`.
- `torch-geometric-temporal`, `torch-geometric`, `wntr`, and `epanet-python-interface` are optional.
- When optional hydraulic or PyG Temporal packages are missing, the code falls back to local implementations.

## Run

From the project root:

```powershell
python main.py
```

You can also tune the batch from the command line:

```powershell
python main.py --benchmark Hanoi_CMH --max-scenarios 20 --epochs 30
```

For a quicker smoke test:

```powershell
python main.py --benchmark Hanoi_CMH --max-scenarios 3 --epochs 5
```

The pipeline will also generate a visual report under:

```text
reports/output/run_YYYYMMDD_HHMMSS/
```

## What to expect when you run it

The current version now produces both terminal logs and saved visual artifacts.

You will see:

- STGCN training loss printed for a few epochs
- Anomaly windows detected from the temporal pressure series
- A leak-isolation recommendation from the report parser
- Before/after pressure comparison after the pipe closure
- Before/after max-flow redirection values
- Before/after MST cost values
- A final summary dictionary logged at the end
- PNG charts and an HTML report saved to the run folder

### Where to see the results

The results are shown in the terminal/log output and in the generated report folder.

Look for lines similar to:

- `Leak triage recommendation`
- `Anomalous windows`
- `Pressure comparison`
- `Flow redirection`
- `MST cost`
- `Pipeline finished successfully`
- `Visual report saved to`

The final run summary is emitted by `main.py` and contains:

- the selected benchmark and input file
- the recommended isolation pipe
- anomaly indices
- pressure comparison metrics
- max-flow routing metrics
- MST cost comparison
- report directory and report file paths

## Are the results graphical?

Yes. The pipeline now saves PNG charts and an HTML report.

Generated visuals include:

- pressure time-series plots
- anomaly score charts
- before/after node pressure bar charts
- operational comparison charts
- before/after network visualizations

Open the HTML file in the run folder to view everything in one place.

## Data

LeakDB assets are expected under:

```text
data/raw/LeakDB/
```

The default demo uses the `Hanoi_CMH` scenario:

- `data/raw/LeakDB/Hanoi_CMH/Scenario-1/Hanoi_CMH_Scenario-1.inp`
- pressure CSVs in `Scenario-1/Pressures/`
- flow CSVs in `Scenario-1/Flows/`
- leak metadata in `Scenario-1/Leaks/`

## Current milestone

This repository is now at a working integrated prototype stage.

It can already:

- ingest a real LeakDB benchmark
- run a lightweight pressure predictor
- close a pipe logically
- recompute hydraulic effects
- compute routing and resilience summaries

It does not yet include:

- a web UI or dashboard
- saved charts
- a fully calibrated EPANET physics engine
- model checkpointing or experiment tracking

## Troubleshooting

- If `python main.py` cannot find data, confirm `data/raw/LeakDB/` exists.
- If PyTorch is missing, install dependencies from `requirements.txt`.
- If optional packages fail to install, the project should still run using the built-in fallbacks.
