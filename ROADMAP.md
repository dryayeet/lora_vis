# Project Expansion Roadmap

Status: exploration notes from a survey of the current PEFT ecosystem (Oct 2026). Nothing here is a commitment. Pick items by letter when ready.

## Research context

- **PEFT ecosystem**: Hugging Face PEFT now supports 40+ adapter types: LoRA plus variants (DoRA, rsLoRA, VeRA, AdaLoRA, LoHa, LoKr, FourierFT, X-LoRA) and non-LoRA methods (IA3, prefix tuning, prompt tuning, P-Tuning). The library also demonstrates adapter lifecycle features this project could mirror: merging (`merge_and_unload`, `add_weighted_adapter`), per-layer rank/alpha patterns, `modules_to_save`, and LoftQ initialization for quantized bases.
- **Transformer Explainer** (Georgia Tech, poloclub.github.io/transformer-explainer): the gold standard for interactive LLM education. Runs GPT-2 in the browser via ONNX with Svelte + D3, live attention maps, and step-by-step QKV math. Our frame-swap pyvis animation is the main UX gap versus that bar.
- **QLoRA internals**: 4-bit NormalFloat quantization, double quantization, and compute dtype switching are well documented. Our memory dashboard only simulates them, so a real quantization demo is possible later.
- **Existing unused hooks in this codebase**: `merge_lora_weights()` and `get_lora_state_dict()` in `src/models/lora.py`, the SVD/low-rank utilities in `src/utils/math_utils.py`, and the dormant `ToyTransformer` class in `src/models/toy_model.py`.

## Expansion menu

### A. Method zoo (2-3 sessions, no new deps)

- Implement LoRA variants in `src/models/lora.py`, each roughly 20-40 lines on the existing `LoRALayer` pattern:
  - DoRA (magnitude + direction decomposition)
  - rsLoRA (alpha / sqrt(r) scaling)
  - LoRA+ (separate learning rates for A and B)
  - VeRA (shared random bases)
  - LoHa (Hadamard product adapters)
  - IA3 and prefix tuning as non-LoRA contrasts
- New "Method Lab" tab: train the same data with 2-3 selected methods side by side, overlay loss curves, compare trainable param counts and delta-W norms.

### B. Adapter lifecycle (1-2 sessions, highest wow per line of code)

- Save/load adapters via `get_lora_state_dict()` (already implemented, unused), with download/upload buttons.
- Merge/unmerge demo via `merge_lora_weights()` (already implemented, unused).
- Multi-adapter composition: train task A and task B adapters, blend them with weights, show how predictions interpolate.

### C. Real transformer (2-4 sessions, largest item)

- Build out `ToyTransformer` as a 2-layer char-level GPT, nanoGPT style, ~100k params, CPU-friendly.
- LoRA on attention Q/K/V/O projections with a target-module picker, mirroring real PEFT semantics.
- Attention heatmap per head, token-level input, next-char prediction output.

### D. Honest data (1 session)

- Ship a small SST-2 subset (~1-2k rows) as a bundled CSV in the repo.
- Real train/val split, accuracy/F1 metrics instead of loss-only.
- Keep the current toy tasks as a "sandbox" option.

### E. Training introspection (1 session)

- Live evolution charts: effective rank of delta-W across epochs (ties into the unused SVD utilities in `math_utils.py`).
- A/B matrix heatmaps per epoch.
- Gradient norm traces (`gradient_norm()` already exists in `math_utils.py`).

### F. Engineering cleanup (under 1 session, do first)

- Keep the `viz/` venv out of version control (already untracked; verify `.gitignore` stays correct).
- Pin `requirements.txt` versions.
- Check the pytest + Streamlit AppTest suite into the repo (harness currently lives outside).
- Add ruff and CI.

## Skipped (decided against for now)

- Custom JS component around vis-network/Cytoscape for smooth animations: rejected, keeping pyvis frame-swap rendering.
- Real-model mode (HF `peft`/`bitsandbytes` with a tiny real LLM behind a flag): not chosen at this time.

## Suggested order

F, then B, then A, then D, then E, then C.

Rationale: F protects everything that follows, B is nearly free given existing code, A maximizes the educational payoff per line, and C is the largest but most rewarding build.
