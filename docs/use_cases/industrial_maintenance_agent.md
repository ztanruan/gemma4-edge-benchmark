# Industrial Maintenance Agent

## Summary

Agent for alarm triage and work-order creation on a production line.

## Why This Matters on Jetson

Lets the production cell reason over local alarms and maintenance knowledge even when the cloud CMMS link is degraded.

## Primary Asset or Workflow

- Organization or site: Lakeside Packaging Line 3
- Domain: high-speed packaging automation
- Asset or workflow: PX-22 palletizer wrist
- Key issue: alarm 44 overcurrent during startup after a gripper swap

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
