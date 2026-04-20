# Caltech 256 Closed-Set Image Classification

## Summary

Balanced 500-image closed-set classification benchmark built from a staged subset of Caltech 256 object categories.

## Why This Matters on Jetson

Measures larger-object visual recognition quality on more natural images while keeping the label space explicit and reviewable.

## Benchmark Semantics

- This is a closed-set image classification benchmark.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
