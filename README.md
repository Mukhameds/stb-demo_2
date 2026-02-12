
---

# STB Planning Core

### A Causal, Signal-Based Decision Architecture (Non-Neural)

This repository demonstrates a structural alternative to neural-network-based decision systems.

It implements a **modular signal architecture** capable of:

* uncertainty-aware decision making
* explicit “wait to gather information” behavior
* causal intervention (`do()`)
* counterfactual reasoning
* online adaptive learning
* learned average causal effect (ACE) estimation

This is not a neural network.
This is not reinforcement learning as usually implemented.
This is a transparent structural decision core.

---

# Strategic Context

Modern AI systems are:

* opaque (LLMs, deep RL)
* difficult to audit
* hard to causally reason about
* not structurally safe by design

This demo explores a different direction:

A lightweight, modular, inspectable decision engine
with built-in causal structure and intervention logic.

The long-term vision:
a deployable adaptive decision core for embedded agents, robotics, and persistent AI systems.

---

# Core Architectural Idea

Decision is not a monolithic function.

Decision is the result of competing **signal modules**.

Each action receives weighted contributions from:

* Goal
* Emotion
* Memory
* Instinct
* Logic
* Will
* SafetyPrior

The system computes logits for actions and converts them to probabilities via softmax.

---

# Key Concept: WAIT as a Planning Operator

Most reactive systems choose between actions immediately.

This architecture introduces:

WAIT — a structural planning action.

WAIT means:

> Do not commit yet.
> Gather better information.
> Then decide.

This transforms the system from reactive → deliberative.

WAIT is gated by:

1. VOI (Value of Information heuristic)
2. Learned ACE (Average Causal Effect)

The system learns when WAIT actually improves outcomes.

---

# Terminology (System Vocabulary)

### TrueRisk

Hidden ground-truth state of environment.

### Obs1 / Obs2

Two-stage observation:

* Obs1: noisy first observation
* Obs2: refined observation after WAIT

### Rel1 / Rel2

Reliability of observation stages.

### Gains

Adaptive weights for signal modules:

* Goal
* Emotion
* Memory
* Instinct
* Logic
* Will

### Memory (avoid-only)

Stores penalties for catastrophic BUY in risky regimes.
Has:

* cap
* decay

### SafetyPrior

Non-learned structural prior to prevent absurd early behavior.

### CausalLearner

Learns:

ACE ≈ E[Reward | do(A1 = WAIT)]
− E[Reward | policy without Logic]

Uses context backoff keys to avoid sparse regimes.

### SCM (Structural Causal Model)

The environment and policy are explicitly modeled as:

Exogenous variables:

* TrueRisk
* Emotion
* Rel1, Rel2
* Noise

Endogenous:

* Obs1
* Obs2
* A1
* A2
* Final
* Reward

Interventions are performed explicitly via `do()`.

---

# What Makes This Different

This system can:

1. Disable Logic and show degradation.
2. Intervene on A1 directly (`do(A1=BUY)`).
3. Intervene on world reliability (`do(Rel1=0.95)`).
4. Compare counterfactual outcomes under the same latent U.

This is causal reasoning, not pattern fitting.

---

# Demonstrated Effects

After training:

* Catastrophe rate in TRUE_HIGH decreases significantly.
* WAIT usage increases selectively in high-uncertainty regimes.
* Logic gain increases adaptively.
* Learned ACE becomes positive in boundary conditions.
* Disabling Logic measurably breaks behavior.

Two canonical demonstrations:

### A) Catastrophe Avoidance

WAIT prevents BUY under misleading first observation.

### B) Profit Rescue

WAIT prevents premature RUN near decision boundary.

Both are verified via structural counterfactual comparison.

---

# Why This Matters for YC

This is not about trading simulation.

This is about proving:

1. A non-neural decision architecture can be:

   * modular
   * inspectable
   * causally analyzable

2. Planning (WAIT) can be:

   * learned structurally
   * evaluated causally
   * proven necessary via intervention

3. Safety-critical behavior can emerge without deep networks.

This architecture is:

* CPU-only
* lightweight
* deterministic in evaluation
* fully auditable
* extensible

---

# Long-Term Direction

This demo is the atomic core of a broader architecture concept:

A persistent, modular AI core capable of:

* long-term memory
* adaptive planning
* explicit intervention reasoning
* embedded deployment
* agent autonomy without black-box models

This is a structural foundation for:

* robotics decision layers
* embedded autonomous agents
* safety-aware planning systems
* personal persistent AI cores

---

# What This Is Not

* Not AGI
* Not a trading bot
* Not a neural network competitor benchmark
* Not a full RL framework

It is a structural proof-of-concept of a causal planning core.

---

# How to Run

```
go run main.go
```

The program:

1. Prints pre-training policy stats.
2. Trains for 1400 episodes.
3. Shows adaptive gain evolution.
4. Displays memory and causal tables.
5. Runs deterministic DO-CHECK.
6. Executes full SCM + counterfactual demo.

---

# Why This Repo Exists

To demonstrate that:

Causal, modular, signal-based decision systems
are viable, inspectable, and extensible.

This is an engineering foundation, not a paper experiment.

---

