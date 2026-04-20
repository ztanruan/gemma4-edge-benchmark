# Multilingual Image Text Extraction

## Summary

Balanced 550-image multilingual text-in-image extraction benchmark built from a staged subset of the multilingual-image-text-translation dataset.

## Why This Matters on Jetson

Measures how well Gemma 4 reads visible text across scripts and languages on-device, which matters for signage, field documents, multilingual labels, and mixed-language edge workflows.

## Benchmark Semantics

- This is a closed-set image classification benchmark.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
