# Field-Service Copilot

## Summary

Assistant for technicians servicing remote commercial equipment with intermittent connectivity.

## Why This Matters on Jetson

Supports technicians on-site without waiting on cloud latency and keeps customer records local when network quality is poor.

## Primary Asset or Workflow

- Organization or site: Meridian Cold Storage, Dock 4
- Domain: refrigeration field service
- Asset or workflow: ThermoGrid R9 controller
- Key issue: intermittent Modbus CRC errors after a controller update to build 3.14.2

## Benchmark Semantics

- Scenario connectivity describes the business environment being simulated.
- Execution mode in this suite is currently `mocked`, which means tools return deterministic local fixtures.
- Context source is synthetic local text tailored to each domain.
