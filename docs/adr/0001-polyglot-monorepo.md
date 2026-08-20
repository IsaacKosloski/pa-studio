# ADR 0001: Polyglot Monorepository
---
## Status
**<span style="color: #1a7f37">Accepted</span>**

## Context
A polyglot product with multiple phases and layers, which must serve as a unified, cohesive showcase and be easily reproducible.

## Decision
Adopt a polyglot monorepository with top-level directories partitioned by component.

## Considered Alternatives
(a) One repository per phase/component: **<span style="color: #cf222e">rejected</span>**, as it fragments the cohesive showcase narrative;
(b) Git submodules: **<span style="color: #cf222e">rejected</span>**, due to high complexity and poor cloning experiences.

## Consequences
- <span style="color: #0969da">Positives</span>: Unified narrative, atomic refactoring across components;
- <span style="color: #bc4c00">Negatives</span>: Repository size increases, CI needs to be component-aware, and large binaries require careful management.
