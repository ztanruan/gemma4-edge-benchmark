# CIFAR-10 Closed-Set Image Classification

## Summary

Balanced 500-image closed-set classification benchmark built from the CIFAR-10 test split.

## Why This Matters on Jetson

Measures how Gemma 4 performs on small, low-resolution visual inputs where edge devices may need lightweight visual triage or object recognition.

## Benchmark Semantics

- This is a closed-set image classification benchmark.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
