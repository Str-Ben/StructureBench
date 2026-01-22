# StructureBench

Benchmarking structured outputs for text/VL models with centralized prompt loaders and dataset-specific evaluators.

## Environment & Dependencies
- Python 3.10 recommended; typical runs use a CUDA GPU (install a matching GPU wheel for torch/transformers).
- Core runtime: `torch` (GPU build), `transformers`, `accelerate`, `datasets`, `tqdm`, `filelock`, `sentencepiece`, `pillow` (for VLM).
- XGrammar constrained decoding (optional): `xgrammar` (install from PyPI or `pip install -e xgrammar` to use the bundled source).
- BFCL extras (imports needed even in prompt-mode): `tree_sitter`, `tree_sitter_java`, `tree_sitter_javascript`, `anthropic`, `cohere`, `openai`, `google-genai`, `mistralai`, `writerai`, `qwen-agent`, `datamodel-code-generator`, `boto3`.
- Install example (CUDA 11.8 wheel; adjust for your CUDA/toolkit):
  ```bash
  # torch GPU wheel (pick the right cuXXX index for your driver)
  pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
  pip install transformers==4.48.3 tokenizers==0.21.0 huggingface-hub==0.26.2 accelerate==1.12.0 datasets sentencepiece pillow filelock tqdm
  pip install tree_sitter==0.21.3 tree_sitter_java==0.21.0 tree_sitter_javascript==0.21.0
  pip install anthropic cohere openai google-genai mistralai writerai qwen-agent datamodel-code-generator boto3
  ```

## Generation (text)
Module: `data/text_models_generate.py`

CLI arguments:
- `--model_name_or_path` (str, required)
- `--device` (`cuda`|`cpu`, default `cuda`)
- `--no_trust_remote_code` (flag)
- `--torch_dtype` (default `float16`)
- `--attn_implementation` (str, optional)
- `--max_new_tokens` (int, default 256)
- `--temperature` (float, default 0.2)
- `--top_p` (float, default 1.0)
- `--repetition_penalty` (float, default 1.1)
- `--frequency_penalty` (float, default 0.2)
- `--dataset` (str, required; registered in `data/prompts.py`)
- `--data_file` (optional override)
- `--split` (default `test`)
- `--num_samples` (int, per-dataset semantics; BFCL interprets as per-category cap)
- `--output_file` (str, required)
- `--prompt_kwargs` (JSON string forwarded to prompt loader)
- Defaults: loader full-scope behavior (e.g., BFCL) when not set.
- `--mode` (`prompt`|`xgrammar`|`guidance`|`outlines`|`llama_cpp`, default `prompt`)
- `--constraint_template` (path to `.json`/`.cfg`/`.ebnf`/`.regex` constraints; used by xgrammar and other constrained backends)
- `--think` (`off`|`on`, Qwen3 think handling)
- `--append_output` (flag; default overwrite)
- `--store_prompt` (flag; default off)
- `--verbose` / `--quiet` (default verbose on; use `--quiet` to disable)

Examples:
```bash
# Generic dataset
python -m data.text_models_generate \
  --dataset humaneval \
  --model_name_or_path <your_model> \
  --output_file outputs/humaneval_pred.jsonl \
  --num_samples 10 --store_prompt

# BFCL single-turn prompt-mode (full scope, 2 per category)
python -m data.text_models_generate \
  --dataset bfcl \
  --model_name_or_path facebook/opt-125m \
  --output_file outputs/bfcl_pred.jsonl \
  --num_samples 2 \
  --prompt_kwargs "{\"test_categories\":[\"simple_python\",\"simple_java\",\"simple_javascript\",\"multiple\",\"parallel\",\"parallel_multiple\",\"live_simple\",\"live_multiple\",\"live_parallel\",\"live_parallel_multiple\"]}" \
  --store_prompt --device cpu

# FOLIO (CoT + 1-shot prompt)
python -m data.text_models_generate \
  --dataset folio \
  --model_name_or_path <your_model> \
  --output_file outputs/folio_pred.jsonl \
  --split validation \
  --num_samples 5 \
  --store_prompt
```

### Constrained generation (xgrammar)
Dependencies:
- Install `xgrammar` (PyPI) or the bundled source (`pip install -e xgrammar`).

Usage:
- Add `--mode xgrammar` and `--constraint_template <path>`.
- Supported templates: `.json` (JSON Schema), `.ebnf` (EBNF grammar), `.regex`.
- Templates live under `constraint_templates/`.

Example:
```bash
python -m data.text_models_generate \
  --dataset text2sql \
  --model_name_or_path <your_model> \
  --output_file outputs/text2sql_pred.jsonl \
  --mode xgrammar \
  --constraint_template constraint_templates/sql/sql.ebnf
```

## Trace generation
Trace scripts under `trace_generator/` run greedy stepwise decoding with xgrammar masks and log per-token decisions to JSONL.

Example:
```bash
python -m trace_generator.humaneval_trace_generate \
  --model_name_or_path <your_model> \
  --num_samples 2 \
  --trace_file trace_generator/logs/humaneval_trace.jsonl
```

Available scripts:
- `trace_generator.humaneval_trace_generate`
- `trace_generator.amr_trace_generate`
- `trace_generator.planetarium_trace_generate`
- `trace_generator.latexocr_trace_generate`
- `trace_generator.smileseval_trace_generate`
- `trace_generator.text2sql_trace_generate`
- `trace_generator.bfcl_trace_generate`

Common flags:
- `--data_file`, `--split`, `--num_samples`
- `--max_new_tokens`, `--temperature`, `--top_p`
- `--repetition_penalty`, `--frequency_penalty`
- `--constraint_template`, `--trace_file`, `--repeat_window`
- plus model flags (`--device`, `--torch_dtype`, `--attn_implementation`, `--no_trust_remote_code`)

## Evaluation
Use dataset-specific evaluators under `eval/`. Generic fallback: `eval/x_eval.py`.

### BFCL single-turn (prompt-mode)
Module: `eval/bfcl_eval`

CLI arguments:
- `--pred_file` (required)
- `--summary_file` (optional; defaults to `eval/bfcl_eval_summary.json`)
- `--data_root` (default `D:\StructureBench\datasets\bfcl_eval\data`)
- `--test_categories` (optional list/group; always limited to allowed scope)
- `--num_samples` (per-category cap; 0 = all)
- `--underscore_to_dot` (flag to normalize function names)
- `--verbose` / `--quiet` (default verbose on)

Example:
```bash
python -m eval.bfcl_eval \
  --pred_file outputs/bfcl_pred.jsonl \
  --summary_file eval/bfcl_eval_summary.json \
  --num_samples 2
```

### FOLIO (Prover9)
Module: `eval/folio_eval.py`

Example:
```bash
python -m eval.folio_eval \
  --pred_file outputs/folio_pred.jsonl \
  --output_file eval/folio_eval.jsonl \
  --summary_file eval/folio_eval_summary.json \
  --split validation
```
Defaults: `data_file` -> `datasets/FOLIO/folio_v2_validation.jsonl` (choose train/validation by `--split`). The evaluator auto-sets `PROVER9`/`MACE4` to the bundled binaries under `eval/prover9/bin`.

### Generic alignment check
```bash
python -m eval.x_eval \
  --dataset <name> \
  --pred_file <pred.jsonl> \
  --output_file eval/<name>_eval.jsonl \
  --summary_file eval/<name>_summary.json \
  --num_samples 10
```

### Other dataset-specific evaluators
See `eval/*_eval.py` for per-dataset flags; most accept `--pred_file`, `--output_file`, `--data_file`, `--split`, `--num_samples`, `--verbose`.

## Files of Interest
- `common/`: shared CLI/model helpers (`cli.py`, `text_model_utils.py`).
- `data/`: centralized prompt loaders (`prompts.py`, `bfcl_prompts.py`), generic generators (`text_models_generate.py`, `vl_models_generate.py`).
- `eval/`: per-dataset evaluators and logging (`bfcl_eval/`, `x_eval.py`, `eval_logging.py`).
