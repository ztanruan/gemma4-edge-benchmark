# On-Prem Customer Support Assistant

## Summary

Support assistant for private tenant and order data with local document grounding.

## Why This Matters on Jetson

Keeps customer tickets and tenant configuration details on-prem while still accelerating case handling.

## Primary Asset or Workflow

- Organization or site: Nimbus Secure Networking
- Domain: enterprise network security support
- Asset or workflow: tenant domain migration flow
- Key issue: users lose SSO after the tenant primary domain is changed

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
