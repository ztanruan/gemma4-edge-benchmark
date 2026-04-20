# Vehicle or Robot Operator Assistant

## Summary

Operator-facing assistant for local diagnostics, safety instructions, and service routing on the machine edge.

## Why This Matters on Jetson

Provides low-latency guidance directly on the vehicle or robot even with no WAN connectivity.

## Primary Asset or Workflow

- Organization or site: Portside Logistics Yard
- Domain: autonomous yard operations
- Asset or workflow: YardBot T7 tug
- Key issue: E214 steering encoder disagreement after a curb strike

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
