
---

# STB Time + Planning Demo

### Causal, Modular Decision Core (Non-Neural)

---

## Overview

This repository contains a minimal structural decision system implemented in Go.

The system:

* makes decisions under uncertainty,
* can choose to delay action (WAIT),
* learns when delaying improves outcomes,
* supports structural causal interventions (`do()`),
* allows deterministic replay for counterfactual comparison.

This is a modular decision core.
It is not a neural network.

---

## Problem Addressed

Reactive systems act immediately based on noisy observations.

In many environments:

* observations are unreliable,
* acting too early causes catastrophic loss,
* waiting can improve decision quality.

The core problem demonstrated here:

> When should a system act immediately, and when should it wait to gather more reliable information?

This demo implements that capability structurally.

---

## Environment Model

The environment contains:

* `TrueRisk` — hidden ground truth
* `Obs1` — initial observation (noisy, reliability = Rel1)
* `Obs2` — second observation (after WAIT, higher reliability = Rel2)

Available actions:

* BUY
* RUN
* WAIT

WAIT enables access to Obs2 before committing to a final action.

---

## Decision Architecture

Each action is evaluated by modular components:

* Goal
* Emotion
* Memory
* Instinct
* Logic
* Will
* SafetyPrior

Each module produces a logit contribution.

Final decision probabilities are computed via softmax.

There is no backpropagation, no hidden layers, no gradient descent.

---

## WAIT as Planning Operator

WAIT is not a random action.

It is activated when:

1. Value of Information (VOI) is high
2. Learned Average Causal Effect (ACE) of WAIT is positive

The system learns when WAIT causally improves reward.

This converts a purely reactive system into a two-step planning system.

---

## Learning

The system adapts:

* Module gains evolve based on reward.
* Memory stores avoid-only penalties for catastrophic actions.
* CausalLearner estimates ACE for WAIT in specific contexts.

Learning is local and structural.

---

## Structural Causal Model (SCM)

The demo includes explicit SCM definitions:

Exogenous variables:

* TrueRisk
* Emotion
* Rel1
* Rel2
* Noise1
* Noise2

Endogenous variables:

* Obs1
* Obs2
* A1 (first-stage action)
* A2 (second-stage action)
* Final
* Reward

The system supports:

* `do(Logic=0)`
* `do(A1=BUY)`
* `do(Rel1=0.95)`
* counterfactual evaluation under fixed latent variables

This allows causal analysis of the decision process.

---

## Demonstrated Behaviors

After training:

* Catastrophic BUY in high-risk regimes decreases.
* WAIT is used primarily in high-uncertainty contexts.
* Logic gain increases when planning is beneficial.
* Disabling Logic degrades performance.
* WAIT shows positive learned ACE in boundary conditions.

Two canonical cases are shown:

### 1. Catastrophe Avoidance

WAIT prevents committing to BUY under misleading observation.

### 2. Profit Rescue

WAIT prevents premature RUN near risk boundary.

Both are validated with structural counterfactual comparison.

---

## What This Demo Is

* A minimal planning core
* A modular decision architecture
* A structural causal reasoning demonstration
* A deterministic replay + intervention testbed
* A non-neural alternative for safety-critical decision logic

---

## Technical Properties

* Deterministic replay mode
* Explicit gain adaptation
* Avoid-only memory mechanism
* Softmax action selection
* Two-stage observation model
* Explicit causal graph
* Intervention and counterfactual support

---

## How to Run

```
go run main.go
```

Execution flow:

1. Show greedy stats before training
2. Train for 1400 episodes
3. Show gain evolution
4. Display memory + ACE tables
5. Execute deterministic DO-CHECK
6. Run full SCM + counterfactual analysis

---

## Perspective

This demo isolates and validates one capability:

> A modular system can learn when delaying action causally improves outcome.

The architecture can be extended toward:

* multi-step planning
* embedded control systems
* robotics decision layers
* interpretable AI cores
* persistent agent systems

It is intended as a minimal, inspectable foundation for larger decision architectures.

---
