# Supplementary Materials for the IJCAI 2026 Paper StructureBench: A Unified Benchmark Suite for Multi-Scenario Structured Generation Tasks with On-Device Models


## Detailed Description of Dataset

| Dataset | Input Modality / Description | Expected Output | Size (Samples) |
| --- | --- | --- | --- |
| BFCL | Natural language, tool-calling queries | JSON (Function call) | 5.1k |
| Hermes-FC | Natural language, dialogue history + JSON Schema | JSON (Information extraction) | 1.2k |
| Cheque | Images of handwritten bank checks | JSON (Structured extraction) | 2.8k |
| JSONSchemaBench | Textual prompts with specific JSON Schemas | Schema-compliant JSON | 9.6k |
| Ape210k | Natural language, primary school math word problems | Mathematical formulas (Calculation process) | 199.0k |
| LaTeX-OCR | Images of handwritten mathematical expressions | LaTeX code | 96.7k |
| HumanEval | Natural language, docstrings and function headers | Python code | 0.2k |
| Text2SQL | Natural language + SQL schema context | SQL query statements | 105.9k |
| SMILES-Eval | Natural language, descriptions of molecular structures | SMILES (Chemical notation) | 0.17k |
| AMR-3-parsed | Natural language, news text | PENMAN (AMR graph notation) | 55.6k |
| Planetarium | Natural language, planning scenarios | PDDL (Planning domain language) | 15.9k |

The StructureBench suite spans 11 datasets across six diverse domains to evaluate the multi-scenario structured generation capabilities of on-device models. To ensure scientific rigor and minimize evaluation bias, we implement a standardized assessment protocol. First, a robust post-processing pipeline is applied to strip redundant whitespace, comments, and formatting artifacts, preventing trivial syntax variations from confounding the assessment of structural reliability. Second, task correctness is determined through functional equivalence using mature, industry-standard parsers, such as SQLGlot for SQL queries, Sympy for mathematical formulas, and RDKit for SMILES notations. This approach prioritizes semantic consistency over exact string matching, correctly identifying valid but varied expressions. Finally, to mitigate stylistic misalignment and guide models with limited instruction-following capacity, each sample-level prompt includes a representative few-shot example along with explicit ground-truth grammar specifications. These measures collectively ensure that observed performance differences reflect the intrinsic reasoning and structural alignment capabilities of the models.

## Detailed Results of the Experiment

Table 1 extends the model-scaling analysis beyond the on-device range, showing that larger models obtain higher SBR and lower CIR under the same constrained-decoding setting. Table 2 isolates the effect of visual input on structured-output accuracy, where the identical scores suggest limited image-feature influence in this setting. Table 3 reports XGrammar efficiency on a JSON grammar task, showing a large reduction in token-mask cache memory with negligible TPOT change.

### Table 1. Diagnostic Scaling Experiment

| Model | SBR | CIR |
| --- | --- | --- |
| Qwen3-8B | 2.99% | 7.20% |
| Qwen3-32B | 34.0% | 2.40% |
| Qwen2.5-72B-Instruct | 51.5% | 1.20% |

### Table 2. Influence of Image Features

| Setting | Acc. |
| --- | --- |
| LLM | 79.2% |
| VLM w/o image | 79.2% |
| VLM w/ blank image | 79.2% |

### Table 3. XGrammar Efficiency Check

| Metric | w/o XGrammar | w/ XGrammar |
| --- | --- | --- |
| Token-mask cache | 160 MB | 0.46 MB |
| TPOT (batch size 1) | 6.2 ms | 6.3 ms |

TPOT denotes time per output token.

This section provides a granular breakdown of the experimental results across all evaluated models, datasets, and constrained decoding frameworks. The statistics presented in Table 4 encompass the four decoupled metrics defined in the metrics section: SFR, SBR, cSBR, and CIR. Notably, a significant portion of the results for SBR and cSBR are recorded as 0.00%, particularly in "Meaning-bound" tasks (e.g., AMR and Text2SQL) or "Exception" scenarios (e.g., Python generation). These zero values signify that constrained decoding provides little marginal gain in semantic correctness over the prompt-only baseline in these instances. For symbolic reasoning tasks, while constraints may successfully enforce syntactic validity (evidenced by non-zero SFR), they fail to resolve the underlying semantic bottlenecks, resulting in zero semantic bonus. In contrast, for open-ended code generation, the zero values often reflect cases where hard constraints induce severe repetition or mask correct reasoning paths, leading to a saturation or even degradation of model performance.

### Table 4. SFR/SBR/cSGR/CIR Ratio for Different Datasets, Models, and Decoders (%)

| Dataset | Model | Constraint | SFR | SBR | cSGR | CIR |
| --- | --- | --- | --- | --- | --- | --- |
| amr | Llama-3.2-1B-Instruct | outlines | 0.00% | 0.00% | 0.00% | 2.26% |
| amr | Llama-3.2-1B-Instruct | xgrammar | 0.00% | 0.00% | 0.00% | 8.47% |
| amr | MiniCPM4-0.5B | outlines | 0.00% | 0.60% | 0.00% | 2.67% |
| amr | MiniCPM4-0.5B | xgrammar | 0.00% | 0.60% | 0.00% | 4.32% |
| amr | Qwen3-0.6B | outlines | 0.00% | 0.60% | 0.00% | 57.06% |
| amr | Qwen3-0.6B | xgrammar | 0.00% | 0.00% | 0.00% | 0.00% |
| amr | Qwen3-8B | outlines | 0.00% | 2.80% | 0.00% | 37.88% |
| amr | Qwen3-8B | xgrammar | 0.00% | 0.00% | 0.00% | 0.20% |
| ape21 | Llama-3.2-1B-Instruct | outlines | 98.67% | 0.00% | 0.67% | 0.00% |
| ape21 | Llama-3.2-1B-Instruct | xgrammar | 84.67% | 0.00% | 0.67% | 0.33% |
| ape21 | MiniCPM4-0.5B | outlines | 28.57% | 0.00% | 0.00% | 0.00% |
| ape21 | MiniCPM4-0.5B | xgrammar | 42.86% | 0.00% | 0.00% | 0.00% |
| ape21 | Qwen3-0.6B | outlines | 30.00% | 0.67% | 0.67% | 34.33% |
| ape21 | Qwen3-0.6B | xgrammar | 34.00% | 0.33% | 0.33% | 26.00% |
| ape21 | Qwen3-8B | outlines | 29.67% | 0.33% | 0.00% | 20.00% |
| ape21 | Qwen3-8B | xgrammar | 33.33% | 0.67% | 1.00% | 15.52% |
| bfcl | Llama-3.2-1B-Instruct | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| bfcl | Llama-3.2-1B-Instruct | xgrammar | 0.00% | 0.00% | 0.00% | 0.89% |
| bfcl | MiniCPM4-0.5B | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| bfcl | MiniCPM4-0.5B | xgrammar | 0.63% | 0.00% | 0.00% | 5.04% |
| bfcl | Qwen3-0.6B | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| bfcl | Qwen3-0.6B | xgrammar | 94.81% | 0.00% | 10.89% | 0.00% |
| bfcl | Qwen3-8B | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| bfcl | Qwen3-8B | xgrammar | 0.63% | 0.00% | 0.00% | 0.00% |
| chequessample | Qwen3-VL-2B-Instruct | outlines | 14.50% | 0.50% | 0.50% | 0.00% |
| chequessample | Qwen3-VL-8B-Instruct | outlines | 22.00% | 9.00% | 4.00% | 0.00% |
| hermes | Llama-3.2-1B-Instruct | guidance | 4.40% | 39.60% | 2.60% | 0.00% |
| hermes | Llama-3.2-1B-Instruct | xgrammar | 4.40% | 30.00% | 2.20% | 0.00% |
| hermes | MiniCPM4-0.5B | guidance | 0.00% | 4.60% | 0.00% | 0.00% |
| hermes | MiniCPM4-0.5B | xgrammar | 0.00% | 9.80% | 0.00% | 0.00% |
| hermes | Qwen3-0.6B | guidance | 0.00% | 5.40% | 0.00% | 0.00% |
| hermes | Qwen3-0.6B | xgrammar | 0.00% | 5.60% | 0.00% | 0.00% |
| hermes | Qwen3-8B | guidance | 0.00% | 4.40% | 0.00% | 0.00% |
| hermes | Qwen3-8B | outlines | 0.00% | 3.60% | 0.00% | 0.00% |
| hermes | Qwen3-8B | xgrammar | 0.00% | 3.80% | 0.00% | 0.00% |
| hermes | openPangu-Embedded-1B | guidance | 0.00% | 16.80% | 0.00% | 0.00% |
| hermes | openPangu-Embedded-1B | outlines | 0.00% | 22.60% | 0.00% | 0.00% |
| hermes | openPangu-Embedded-1B | xgrammar | 0.00% | 23.60% | 0.00% | 0.00% |
| humaneval | Llama-3.2-1B-Instruct | outlines | 0.61% | 2.44% | 0.00% | 7.32% |
| humaneval | Llama-3.2-1B-Instruct | xgrammar | 1.22% | 0.61% | 0.00% | 23.78% |
| humaneval | MiniCPM4-0.5B | outlines | 12.80% | 1.83% | 0.00% | 100.00% |
| humaneval | MiniCPM4-0.5B | xgrammar | 7.32% | 1.22% | 0.00% | 98.33% |
| humaneval | Qwen2.5-Coder-0.5B-Instruct | outlines | 1.25% | 1.88% | 0.62% | 13.75% |
| humaneval | Qwen2.5-Coder-0.5B-Instruct | xgrammar | 1.25% | 1.25% | 1.25% | 11.25% |
| humaneval | Qwen3-0.6B | outlines | 10.98% | 0.61% | 0.00% | 32.89% |
| humaneval | Qwen3-0.6B | xgrammar | 11.59% | 1.22% | 0.61% | 5.26% |
| humaneval | Qwen3-8B | outlines | 0.00% | 1.25% | 0.00% | 6.25% |
| humaneval | Qwen3-8B | xgrammar | 0.00% | 0.00% | 0.00% | 5.62% |
| humaneval | deepseek-coder-1.3b-instruct | outlines | 0.61% | 1.22% | 0.00% | 68.90% |
| humaneval | deepseek-coder-1.3b-instruct | xgrammar | 0.61% | 1.22% | 0.00% | 47.56% |
| humaneval | openPangu-Embedded-1B | xgrammar | 3.05% | 0.00% | 0.00% | 9.76% |
| jsschemabench | MiniCPM4-0.5B | xgrammar | 5.97% | 57.46% | 4.48% | 5.34% |
| jsschemabench | Qwen3-0.6B | guidance | 5.22% | 32.09% | 4.48% | 17.91% |
| jsschemabench | Qwen3-0.6B | xgrammar | 2.99% | 39.55% | 2.24% | 2.99% |
| jsschemabench | Qwen3-8B | guidance | 1.49% | 18.66% | 1.49% | 2.99% |
| jsschemabench | openPangu-Embedded-1B | xgrammar | 16.42% | 44.78% | 16.42% | 0.75% |
| latexocr | Qwen3-VL-2B-Instruct | outlines | 3.20% | 15.60% | 1.00% | 13.23% |
| latexocr | Qwen3-VL-8B-Instruct | outlines | 3.20% | 24.40% | 1.00% | 7.00% |
| planetarium | Llama-3.2-1B-Instruct | outlines | 50.00% | 0.00% | 0.33% | 0.00% |
| planetarium | Llama-3.2-1B-Instruct | xgrammar | 39.00% | 0.00% | 0.00% | 1.02% |
| planetarium | MiniCPM4-0.5B | outlines | 12.00% | 0.33% | 0.33% | 46.24% |
| planetarium | MiniCPM4-0.5B | xgrammar | 5.00% | 0.00% | 0.00% | 73.48% |
| planetarium | Qwen3-0.6B | outlines | 58.20% | 0.00% | 0.40% | 11.80% |
| planetarium | Qwen3-0.6B | xgrammar | 3.60% | 0.00% | 0.00% | 1.20% |
| planetarium | Qwen3-8B | outlines | 16.00% | 0.40% | 0.00% | 1.00% |
| planetarium | Qwen3-8B | xgrammar | 14.00% | 0.00% | 0.00% | 0.00% |
| smiles | MiniCPM4-0.5B | outlines | 1.80% | 0.00% | 0.00% | 0.00% |
| smiles | MiniCPM4-0.5B | xgrammar | 22.16% | 0.00% | 0.00% | 0.00% |
| smiles | Qwen3-0.6B | outlines | 0.00% | 0.00% | 0.00% | 0.63% |
| smiles | Qwen3-0.6B | xgrammar | 26.95% | 0.00% | 0.60% | 0.00% |
| smiles | Qwen3-8B | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| smiles | Qwen3-8B | xgrammar | 17.96% | 1.20% | 2.99% | 1.22% |
| smiles | openPangu-Embedded-1B | xgrammar | 0.00% | 0.00% | 0.00% | 0.00% |
| text2sql | MiniCPM4-0.5B | outlines | 0.00% | 4.33% | 0.00% | 99.33% |
| text2sql | MiniCPM4-0.5B | xgrammar | 0.00% | 5.67% | 0.00% | 6.33% |
| text2sql | Qwen3-0.6B | outlines | 0.00% | 0.00% | 0.00% | 0.00% |
| text2sql | Qwen3-0.6B | xgrammar | 0.00% | 2.39% | 0.00% | 0.00% |
| text2sql | Qwen3-8B | outlines | 0.00% | 0.00% | 0.00% | 61.00% |
| text2sql | Qwen3-8B | xgrammar | 0.00% | 0.00% | 0.00% | 33.67% |
