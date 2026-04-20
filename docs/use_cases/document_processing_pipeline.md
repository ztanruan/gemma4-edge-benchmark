# Local Document Processing Pipeline

## Summary

Structured extraction and exception routing for invoices and intake documents on the edge.

## Why This Matters on Jetson

Processes sensitive business documents locally and supports site-level automation even when cloud APIs are restricted.

## Primary Asset or Workflow

- Organization or site: Apex Industrial Supplies
- Domain: accounts payable automation
- Asset or workflow: invoice intake document
- Key issue: invoice total mismatches the purchase order beyond the 1.5 percent tolerance

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
