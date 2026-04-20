# Retail and Warehouse Operations Assistant

## Summary

Operations assistant for local inventory exceptions, replenishment, and task triage.

## Why This Matters on Jetson

Lets associates resolve exceptions with low latency at the site edge and continue operating through WAN instability.

## Primary Asset or Workflow

- Organization or site: Harbor North Fulfillment Center
- Domain: warehouse operations
- Asset or workflow: wave 7721 replenishment issue
- Key issue: pickers are hitting an empty forward slot while on-hand inventory still shows positive

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
