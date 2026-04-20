# Healthcare Admin Assistant

## Summary

Assistant for referral, prior-authorization, and scheduling workflows in a private clinical environment.

## Why This Matters on Jetson

Keeps patient-adjacent workflow data on local infrastructure and still supports downtime or intermittent connectivity scenarios.

## Primary Asset or Workflow

- Organization or site: Cedar Grove Orthopedics
- Domain: healthcare administration
- Asset or workflow: lumbar MRI prior authorization
- Key issue: payer requires either six weeks of conservative therapy or documented red-flag symptoms

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
