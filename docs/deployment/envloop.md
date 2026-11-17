# EnvLoop runtime

The EnvLoop fork maintains Python changes on `env-loop-main`, based on
`releases/v0.33.0` (`4ba787ec9959e566d3b587a796dff07673324da5`).
It supports independent Q/K/V and legacy fused QKV LoRA adapters for Qwen3.5,
GLM packed projection adapters, and batch token row mapping for GPU prompt
logprobs, including the fixed token scoring path. Conflicting fused and split
adapters fail before packing. Unsupported non-GPU prompt LoRA implementations
fail explicitly rather than silently applying the wrong sampled row mapping.

EnvHub owns the unified CUDA image. Its build inherits the fixed official
vLLM image and overlays only reviewed Python files from one complete fork
commit. It verifies original Python files and preserves native binaries,
generated version metadata, installed dependencies and entry points. A CPU
builder needs no GPU or CUDA compilation. Model execution still requires a
supported NVIDIA GPU. Changes to native interfaces require a new binary build
and cannot use this overlay route.

Recipe handoff, model selection, middleware, metrics and checkpoint publication
remain in envloop-recipe. Recipes do not modify the installed engine at startup.
`vllm.envloop_runtime.CAPABILITIES` lets a Recipe reject an incompatible runtime.
