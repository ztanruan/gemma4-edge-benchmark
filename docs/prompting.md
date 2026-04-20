# vLLM Chat Prompting Strategy

This suite benchmarks Gemma 4 on Jetson through vLLM's supported `/v1/chat/completions` path.

- Requests are sent as structured chat `messages`, not raw completion prompts.
- `thinking=true` is enabled with `chat_template_kwargs.enable_thinking=true`.
- Prompt token counts are estimated by sending the structured `messages` to vLLM's `/tokenize` endpoint so the real chat template is applied.
- For image scenarios, the user content is interleaved multimodal content with the image item first and the text instruction second, following Gemma multimodal best practice.
- For multimodal chat serving through vLLM, the API server inserts the model-specific image placeholder tokens automatically, so the benchmark does not manually add `<|image|>` to the text content.
- The deployed benchmark profile is text+image only. Audio is intentionally unsupported in this Jetson benchmark kit.
- The baseline image config pins `max_soft_tokens=280`, which is the official Gemma 4 vLLM default.
- The kit also generates `560` and `1120` image configs so you can rerun the same image families and compare vision-budget tradeoffs directly.
- The Gemma 4 reasoning parser surfaces the model's reasoning in the `reasoning` field when available.
- Agent scenarios pass OpenAI-style `tools` schemas to the chat completion request.
- Tool responses are appended as `tool` role messages in the next request turn.
- The benchmark collects outputs and metrics only. It does not auto-score correctness.
