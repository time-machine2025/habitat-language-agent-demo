# Habitat Language Agent Demo

This repository is a lightweight embodied agent demo inspired by Habitat-style language grounding workflows.

It is designed for reviewability rather than simulator fidelity:

- a house-style grid environment with rooms and obstacles
- language instructions about moving objects across rooms
- a planner that outputs high-level steps
- an executor that grounds those steps with BFS path planning
- a benchmark script across several scenarios

## Quick Start

```bash
python3 scripts/run_demo.py
```

Requirements:

- Python 3.10+
- no third-party dependencies for the current demo

You can also force the rule planner:

```bash
python3 scripts/run_demo.py --planner rule
```

Current sample benchmark:

```text
Planner success rate: 1.00 | avg steps: 10.00
Random success rate: 0.00 | avg steps: 60.00
```

## Pipeline

```text
Language instruction
  -> grounded object/receptacle references
  -> structured plan
  -> path planning and execution
  -> success / step metrics
```

## Minimal Verification

```bash
python3 -m unittest discover -s tests -q
```


## What Is Honest About It

- the world is a house-style grid abstraction, not the official Habitat simulator
- the planning interface is shaped like a real embodied pipeline
- the value here is software structure and research communication, not benchmark claims
