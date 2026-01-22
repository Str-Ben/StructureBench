#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOLIO evaluator (Prover9-based) migrated from CRANE.

Metrics:
- Accuracy: program compiles and Prover9-derived answer matches gold (A/B/C).
- Syntax errors: program fails to parse/compile.
- Semantic errors: program compiles but returns no answer or wrong answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None

from eval.eval_logging import append_eval_log


LABEL_TO_ANSWER = {"True": "A", "False": "B", "Uncertain": "C"}
SYMBOL_REPLACEMENTS = {
    "{and}": "\u2227",
    "{or}": "\u2228",
    "{xor}": "\u2295",
    "{not}": "\u00ac",
    "{implies}": "\u2192",
    "{iff}": "\u2194",
    "{forall} ": "\u2200",
    "{exists} ": "\u2203",
    "{forall}": "\u2200",
    "{exists}": "\u2203",
}


def _json_default(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _get_field(item: Any, key: str, default=None):
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _index_predictions(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    pred_map: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for idx, record in enumerate(records):
        sample_id = _get_field(record, "sample_id", None)
        if sample_id is None:
            sample_id = f"idx-{idx}"
        sample_id = str(sample_id)
        if sample_id in pred_map:
            duplicates += 1
            continue
        pred_map[sample_id] = record
    return pred_map, duplicates


def _resolve_sample_id(sample: Mapping[str, Any], idx: int) -> str:
    for key in ("example_id", "id", "idx", "story_id"):
        if key in sample:
            return str(sample[key])
    return f"idx-{idx}"


def _replace_symbols(text: str) -> str:
    for src, dst in SYMBOL_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def _extract_logic_program(prediction: str) -> str:
    if not prediction:
        return ""
    logic = prediction
    if "Predicates:" in logic:
        logic = "Predicates:" + logic.split("Predicates:", 1)[1]
    if "------" in logic:
        logic = logic.split("------", 1)[0]
    if "Note" in logic:
        logic = logic.split("Note", 1)[0]
    return logic.strip()


def _strip_formula_prefix(formula: str) -> str:
    formula = re.sub(r"^\s*(premise|premises|conclusion)\s*:\s*", "", formula, flags=re.IGNORECASE)
    formula = re.sub(r"^\s*[-*\d.()]+[\s:]+", "", formula)
    return formula.strip()


def _replace_word_operators(formula: str) -> str:
    formula = re.sub(r"\bif\s+and\s+only\s+if\b", "{iff}", formula, flags=re.IGNORECASE)
    formula = re.sub(r"\biff\b", "{iff}", formula, flags=re.IGNORECASE)
    formula = formula.replace("<->", "{iff}").replace("<=>", "{iff}")
    formula = formula.replace("->", "{implies}").replace("=>", "{implies}")

    match = re.match(r"^\s*if\s+(.*?)\s*,?\s*then\s+(.*)$", formula, flags=re.IGNORECASE)
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip()
        formula = f"({left}) {{implies}} ({right})"

    formula = re.sub(r"(?<!\{)\bnot\b(?!\})\s*\(", "{not}(", formula, flags=re.IGNORECASE)
    formula = re.sub(r"(?<!\{)\bnot\b(?!\})", "{not}", formula, flags=re.IGNORECASE)
    formula = re.sub(r"(?<!\{)\band\b(?!\})", "{and}", formula, flags=re.IGNORECASE)
    formula = re.sub(r"(?<!\{)\bor\b(?!\})", "{or}", formula, flags=re.IGNORECASE)
    formula = re.sub(r"(?<!\{)\bxor\b(?!\})", "{xor}", formula, flags=re.IGNORECASE)
    formula = re.sub(r"\bthen\b", "", formula, flags=re.IGNORECASE)
    return formula


def _flatten_nested_predicates(formula: str) -> str:
    pattern = re.compile(r"(\b[A-Za-z_]\w*)\(\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\)")
    while True:
        match = pattern.search(formula)
        if not match:
            break
        outer, inner_pred, inner_arg = match.groups()
        if len(inner_arg) > 1:
            replacement_arg = inner_arg
        elif inner_arg.lower() in {"x", "y", "z", "u", "v", "w"}:
            replacement_arg = inner_pred
        else:
            replacement_arg = inner_pred
        formula = formula[: match.start()] + f"{outer}({replacement_arg})" + formula[match.end() :]
    return formula


def _looks_like_formula(formula: str) -> bool:
    if "(" not in formula or ")" not in formula:
        return False
    return bool(re.search(r"[A-Za-z_]\w*\s*\(", formula))


def _sanitize_formula_line(line: str, fallback_original: bool = False) -> Optional[str]:
    original = line.strip()
    if not original:
        return None
    if "::: " in original:
        formula, gloss = original.split(":::", 1)
    elif ":::" in original:
        formula, gloss = original.split(":::", 1)
    else:
        formula, gloss = original, ""
    formula = _strip_formula_prefix(formula)
    formula = _replace_word_operators(formula)
    formula = _flatten_nested_predicates(formula)
    formula = formula.strip().rstrip(".;")
    formula = re.sub(r"\s+", " ", formula).strip()
    if not _looks_like_formula(formula):
        if not fallback_original:
            return None
        fallback_formula = _strip_formula_prefix(original)
        fallback_formula = re.sub(r"\s+", " ", fallback_formula).strip()
        return fallback_formula if fallback_formula else original
    if gloss.strip():
        return f"{formula} ::: {gloss.strip()}"
    return formula


def _postprocess_logic_program(logic_program: str) -> str:
    if not logic_program:
        return logic_program
    lines = logic_program.splitlines()
    output: list[str] = []
    in_premises = False
    in_conclusion = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("predicates:"):
            in_premises = False
            in_conclusion = False
            output.append("Predicates:")
            continue
        if lowered.startswith("premises:"):
            in_premises = True
            in_conclusion = False
            output.append("Premises:")
            continue
        if lowered.startswith("conclusion:"):
            in_premises = False
            in_conclusion = True
            output.append("Conclusion:")
            continue
        if stripped.startswith("------"):
            break
        if in_premises or in_conclusion:
            fixed = _sanitize_formula_line(stripped, fallback_original=in_conclusion)
            if fixed:
                output.append(fixed)
            continue
        output.append(stripped)
    return "\n".join(output)


def _ensure_prover9_env() -> Optional[Path]:
    """Set default PROVER9/MACE4 env vars to the bundled binary if unset."""
    bin_path = Path(__file__).resolve().parent / "prover9" / "bin"
    prover9_bin = bin_path / "prover9"
    mace4_bin = bin_path / "mace4"
    if prover9_bin.exists():
        os.environ.setdefault("PROVER9", str(prover9_bin))
    if mace4_bin.exists():
        os.environ.setdefault("MACE4", str(mace4_bin))
    return bin_path if prover9_bin.exists() else None


class FOL_Parser:
    def __init__(self) -> None:
        self.op_ls = ["\u2295", "\u2228", "\u2227", "\u2192", "\u2194", "\u2200", "\u2203", "\u00ac", "(", ")", ","]
        self.sym_reg = re.compile(r"[^\u2295\u2228\u2227\u2192\u2194\u2200\u2203\u00ac(),]+")
        self.cfg_template = (
            "        S -> F | Q F | '\\u00ac' S | '(' S ')'\n"
            "        Q -> QUANT VAR | QUANT VAR Q\n"
            "        F -> '\\u00ac' '(' F ')' | '(' F ')' | F OP F | L\n"
            "        OP -> '\\u2295' | '\\u2228' | '\\u2227' | '\\u2192' | '\\u2194'\n"
            "        L -> '\\u00ac' PRED '(' TERMS ')' | PRED '(' TERMS ')'\n"
            "        TERMS -> TERM | TERM ',' TERMS\n"
            "        TERM -> CONST | VAR\n"
            "        QUANT -> '\\u2200' | '\\u2203'\n"
        )

    def parse_text_FOL_to_tree(self, rule_str):
        import nltk

        tokens, parsed_fol_str = self.msplit(rule_str)
        cfg_str = self.make_cfg_str(tokens)
        grammar = nltk.CFG.fromstring(cfg_str)
        parser = nltk.ChartParser(grammar)
        try:
            tree = parser.parse_one(tokens)
        except Exception:
            tree = None
        return tree

    def msplit(self, s):
        for op in self.op_ls:
            s = s.replace(op, f" {op} ")
        r = [e.strip() for e in s.split()]
        r = [e.replace("'", "") for e in r]
        r = [e for e in r if e != ""]

        res = []
        cur_str_ls = []
        for e in r:
            if (len(e) > 1) and self.sym_reg.match(e):
                cur_str_ls.append(e[0].upper() + e[1:])
            else:
                if len(cur_str_ls) > 0:
                    res.extend(["".join(cur_str_ls), e])
                else:
                    res.extend([e])
                cur_str_ls = []
        if len(cur_str_ls) > 0:
            res.append("".join(cur_str_ls))

        return res, "".join(r)

    def make_cfg_str(self, token_ls):
        sym_ls = list(set([e for e in token_ls if self.sym_reg.match(e)]))
        sym_str = " | ".join([f"'{s}'" for s in sym_ls]) or "'sym'"
        cfg_str = (
            self.cfg_template
            + f"VAR -> {sym_str}\nPRED -> {sym_str}\nCONST -> {sym_str}"
        )
        return cfg_str

    def find_variables(self, lvars, tree):
        if tree is None or isinstance(tree, str):
            if isinstance(tree, str):
                lvars.add(tree)
            return
        if tree.label() == "VAR":
            lvars.add(tree[0])
            return
        for child in tree:
            self.find_variables(lvars, child)

    def preorder_resolution(self, tree, lvars, consts, preds):
        if tree is None or isinstance(tree, str):
            return
        if tree.label() == "PRED":
            preds.add(tree[0])
            return
        if tree.label() == "TERM":
            sym = tree[0][0]
            if sym in lvars:
                tree[0].set_label("VAR")
            else:
                tree[0].set_label("CONST")
                consts.add(sym)
            return
        for child in tree:
            self.preorder_resolution(child, lvars, consts, preds)

    def symbol_resolution(self, tree):
        lvars, consts, preds = set(), set(), set()
        self.find_variables(lvars, tree)
        self.preorder_resolution(tree, lvars, consts, preds)
        return lvars, consts, preds


class FOL_Formula:
    def __init__(self, str_fol) -> None:
        try:
            self.parser = FOL_Parser()
            tree = self.parser.parse_text_FOL_to_tree(str_fol)
            self.tree = tree
            self.is_valid = tree is not None
            if self.is_valid:
                self.variables, self.constants, self.predicates = self.parser.symbol_resolution(tree)
            else:
                self.variables, self.constants, self.predicates = set(), set(), set()
        except Exception:
            self.tree = None
            self.is_valid = False
            self.variables, self.constants, self.predicates = set(), set(), set()

    def __str__(self) -> str:
        if not self.tree:
            return ""
        _, parsed_fol_str = self.parser.msplit("".join(self.tree.leaves()))
        return parsed_fol_str

    def get_formula_template(self):
        template = self.tree.copy(deep=True) if self.tree else None
        name_mapping = {}
        if template is None:
            self.template_str = ""
            return name_mapping, ""
        for i, f in enumerate(self.predicates):
            name_mapping[f] = f"F{i}"
        for i, f in enumerate(self.constants):
            name_mapping[f] = f"C{i}"
        self._get_formula_template(template, name_mapping)
        _, self.template_str = self.parser.msplit("".join(template.leaves()))
        return name_mapping, self.template_str

    def _get_formula_template(self, tree, name_mapping):
        if tree is None:
            return
        for i, subtree in enumerate(tree):
            if isinstance(subtree, str):
                if subtree in name_mapping:
                    tree[i] = name_mapping[subtree]
            else:
                self._get_formula_template(subtree, name_mapping)


class Prover9_FOL_Formula:
    def __init__(self, fol_formula: FOL_Formula) -> None:
        from ply import lex, yacc

        self.tokens = ["QUANT", "VAR", "NOT", "LPAREN", "RPAREN", "OP", "PRED", "COMMA", "CONST"]

        self.t_QUANT = r"\u2200|\u2203"
        self.t_NOT = r"\u00ac"
        self.t_LPAREN = r"\("
        self.t_RPAREN = r"\)"
        self.t_OP = r"\u2295|\u2228|\u2227|\u2192|\u2194"
        self.t_COMMA = r","

        self.t_VAR = r"|".join(list(fol_formula.variables)) if fol_formula.variables else r"x"
        self.t_PRED = r"|".join(list(fol_formula.predicates)) if fol_formula.predicates else r"PRED"
        self.t_CONST = r"|".join(list(fol_formula.constants)) if fol_formula.constants else r"c0"

        self.precedence = (("left", "OP"), ("right", "NOT"))
        self.t_ignore = " \t"
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, write_tables=False, debug=False)

        self.formula = self.parse(str(fol_formula))

    def t_error(self, t):
        t.lexer.skip(1)

    def p_S_F(self, p):
        """expr : F"""
        p[0] = p[1]

    def p_S_quantified_S(self, p):
        """expr : QUANT VAR expr"""
        if p[1] == "\u2200":
            p[0] = f"all {p[2]}.({p[3]})"
        elif p[1] == "\u2203":
            p[0] = f"some {p[2]}.({p[3]})"

    def p_S_not(self, p):
        """expr : NOT expr"""
        p[0] = f"not ({p[2]})"

    def p_F_not(self, p):
        """F : NOT LPAREN F RPAREN"""
        p[0] = f"not ({p[3]})"

    def p_F_paren(self, p):
        """F : LPAREN F RPAREN"""
        p[0] = p[2]

    def p_F_var(self, p):
        """F : VAR"""
        p[0] = p[1]

    def p_F_op(self, p):
        """F : F OP F"""
        if p[2] == "\u2295":
            p[0] = f"(({p[1]}) & not ({p[3]})) | (not ({p[1]}) & ({p[3]}))"
        elif p[2] == "\u2228":
            p[0] = f"({p[1]}) | ({p[3]})"
        elif p[2] == "\u2227":
            p[0] = f"({p[1]}) & ({p[3]})"
        elif p[2] == "\u2192":
            p[0] = f"({p[1]}) -> ({p[3]})"
        elif p[2] == "\u2194":
            p[0] = f"({p[1]}) <-> ({p[3]})"

    def p_F_L(self, p):
        """F : L"""
        p[0] = p[1]

    def p_L_not(self, p):
        """L : NOT PRED LPAREN TERMS RPAREN"""
        p[0] = f"not {p[2]}({p[4]})"

    def p_L_pred(self, p):
        """L : PRED LPAREN TERMS RPAREN"""
        p[0] = f"{p[1]}({p[3]})"

    def p_TERMS_TERM(self, p):
        """TERMS : TERM"""
        p[0] = p[1]

    def p_TERMS_TERM_TERMS(self, p):
        """TERMS : TERM COMMA TERMS"""
        p[0] = f"{p[1]}, {p[3]}"

    def p_TERM_CONST(self, p):
        """TERM : CONST"""
        p[0] = p[1]

    def p_TERM_VAR(self, p):
        """TERM : VAR"""
        p[0] = p[1]

    def p_error(self, p):
        pass

    def parse(self, s):
        return self.parser.parse(s, lexer=self.lexer)


class FOL_Prover9_Program:
    def __init__(self, logic_program: str, dataset_name: str = "FOLIO") -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.compiles = self.parse_logic_program()

    def parse_logic_program(self):
        try:
            if "Premises:" not in self.logic_program or "Conclusion:" not in self.logic_program:
                return False

            premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
            conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

            premises = premises_string.strip().split("\n")
            conclusion = conclusion_string.strip().split("\n")

            self.logic_premises = [premise.split(":::")[0].strip() for premise in premises if premise.strip()]
            self.logic_conclusion = conclusion[0].split(":::")[0].strip() if conclusion else ""

            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid is False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid is False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except Exception:
            return False

    def execute_program(self):
        try:
            from nltk.inference.prover9 import Prover9Command
            from nltk.sem.logic import NegatedExpression, Expression
        except Exception as exc:  # pragma: no cover - optional dependency
            return None, f"Prover9 unavailable: {exc}"

        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 1000

            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            if result:
                return "True", ""
            negated_goal = NegatedExpression(goal)
            prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
            negation_result = prover.prove()
            if negation_result:
                return "False", ""
            return "Unknown", ""
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def answer_mapping(answer: Optional[str]) -> Optional[str]:
        if answer in LABEL_TO_ANSWER:
            return LABEL_TO_ANSWER[answer]
        return None


def _resolve_data_file(data_file: Optional[str], split: str) -> str:
    if data_file:
        return data_file
    split_key = split.lower()
    if split_key in {"train", "training"}:
        return "datasets/FOLIO/folio_v2_train.jsonl"
    return "datasets/FOLIO/folio_v2_validation.jsonl"


def _evaluate_logic_program(logic_program: str, gold_answer: Optional[str]):
    if not logic_program:
        return {
            "status": "syntax error",
            "compiles": False,
            "answer": None,
            "error_message": None,
        }

    program = FOL_Prover9_Program(logic_program)
    compiles = program.compiles
    if not compiles:
        return {"status": "syntax error", "compiles": False, "answer": None, "error_message": None}

    answer_raw, error_message = program.execute_program()
    answer_label = program.answer_mapping(answer_raw)

    if answer_label is None:
        return {
            "status": "semantic error",
            "compiles": True,
            "answer": None,
            "error_message": error_message,
        }

    status = "correct" if (gold_answer is not None and answer_label == gold_answer) else "semantic error"
    return {
        "status": status,
        "compiles": True,
        "answer": answer_label,
        "error_message": error_message,
    }


def evaluate_sample(
    sample: Mapping[str, Any],
    prediction: Optional[str],
    pred_record: Optional[Mapping[str, Any]] = None,
    sample_id: Optional[str] = None,
    context: Optional[Any] = None,
    **_: Any,
) -> Dict[str, Any]:
    """x_eval-compatible single-sample evaluation."""
    _ensure_prover9_env()
    del pred_record, sample_id, context  # unused in this helper
    gold_label = str(sample.get("label"))
    gold_answer = LABEL_TO_ANSWER.get(gold_label)

    logic_program = _replace_symbols(_postprocess_logic_program(_extract_logic_program(str(prediction or ""))))
    result = _evaluate_logic_program(logic_program, gold_answer)
    return {
        "status": result["status"] if result.get("status") != "correct" else "ok",
        "logic_program": logic_program,
        "compiles": result.get("compiles"),
        "answer": result.get("answer"),
        "gold_answer": gold_answer,
        "error_message": result.get("error_message"),
    }


def load_folio_dataset(
    dataset: str,
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[Any]:
    del dataset  # unused; required by x_eval signature
    if load_dataset is None:
        raise RuntimeError("datasets is required to load FOLIO records.")
    split = split or "validation"
    data_file = _resolve_data_file(data_file, split)
    ds_dict = load_dataset("json", data_files={split: data_file})
    ds = ds_dict[split]
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        yield ds[idx]


def run_evaluation(args) -> None:
    _ensure_prover9_env()
    if load_dataset is None:
        raise RuntimeError("datasets is required to load FOLIO records.")

    data_file = _resolve_data_file(args.data_file, args.split)
    ds_dict = load_dataset("json", data_files={args.split: data_file})
    if args.split not in ds_dict:
        raise ValueError(f"split {args.split!r} not found in data file; available: {list(ds_dict.keys())}")
    ds = ds_dict[args.split]

    pred_map, duplicates = _index_predictions(_read_jsonl(args.pred_file))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    n = len(ds) if not args.num_samples or args.num_samples <= 0 else min(args.num_samples, len(ds))
    correct = 0
    evaluated = 0
    syntax_errors = 0
    semantic_errors = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {"dataset": "folio", "sample_id": sample_id, "status": "missing_pred"}
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            prediction_text = str(_get_field(pred_record, "raw_output", "") or "")
            gold_label = str(sample.get("label"))
            gold_answer = LABEL_TO_ANSWER.get(gold_label)

            logic_program = _replace_symbols(_postprocess_logic_program(_extract_logic_program(prediction_text)))
            evaluated += 1

            result = _evaluate_logic_program(logic_program, gold_answer)
            status = result["status"]
            if status == "correct":
                correct += 1
            elif status == "syntax error":
                syntax_errors += 1
            else:
                semantic_errors += 1

            record = {
                "dataset": "folio",
                "sample_id": sample_id,
                "status": status,
                "logic_program": logic_program,
                "compiles": result.get("compiles"),
                "answer": result.get("answer"),
                "gold_answer": gold_answer,
                "error_message": result.get("error_message"),
            }
            if args.store_prediction:
                record["prediction"] = prediction_text
            if args.verbose:
                print(
                    f"[Sample {idx}] id={sample_id} status={status} "
                    f"answer={result.get('answer')} gold={gold_answer} compiles={result.get('compiles')}"
                )
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    accuracy = correct / evaluated if evaluated else None
    if evaluated > 0:
        print("=" * 80)
        print(f"Checked samples (with predictions): {evaluated}")
        print(f"Accuracy: {correct}/{evaluated}" + (f" = {accuracy:.2%}" if accuracy is not None else ""))
        print(f"Syntax errors: {syntax_errors}")
        print(f"Semantic errors: {semantic_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="folio",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_errors,
        semantic_errors=semantic_errors,
        total_samples=n,
        evaluated_samples=evaluated,
        missing_pred=missing_pred,
        duplicates=duplicates,
        eval_script=Path(__file__).name,
        extra={"data_file": data_file, "split": args.split, "num_samples": args.num_samples},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FOLIO predictions with Prover9.")
    parser.add_argument("--pred_file", type=str, required=True, help="JSONL file produced by text_models_generate.py.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write per-sample evaluation JSONL.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Optional dataset file path; defaults to datasets/FOLIO mirror.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split; determines default file selection.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--append_output",
        action="store_true",
        help="Append to output_file instead of overwriting.",
    )
    parser.add_argument(
        "--store_prediction",
        action="store_true",
        help="Store prediction text in the output JSONL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample evaluation status.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
