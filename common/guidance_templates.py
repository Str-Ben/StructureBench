import os
import sys
import time

import guidance
from guidance import gen, select, one_or_more, zero_or_more, optional

# ===========================================================================================================
# python for humaneval (stabilized)
# ===========================================================================================================

# Debug switch (avoid massive prints that make decoding look like an infinite loop)
_GUIDE_DBG = os.environ.get("CONSDE_GUIDANCE_DEBUG", "0") == "1"


def _dbg(msg: str):
    if not _GUIDE_DBG:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[guidance_templates {ts}] {msg}", file=sys.stderr, flush=True)


# Hard limits to keep grammar tractable for guidance
_MAX_BLOCK_DEPTH = 10
_MAX_EXPR_DEPTH = 10
_MAX_STMTS_PER_BLOCK = 6
_MAX_BINOPS = 4
_MAX_TRAILERS = 2
_MAX_CALL_ARGS = 3

_NEWLINE = "\n"
_WS = " "
_COLON = ":"

_INDENTS = {i: "    " * i for i in range(1, _MAX_BLOCK_DEPTH + 1)}


# ==========================================
# Terminals
# ==========================================

@guidance(stateless=True)
def _NAME(lm):
    return lm + gen(regex=r"[A-Za-z_][A-Za-z0-9_]*")


@guidance(stateless=True)
def _NUMBER(lm):
    return lm + gen(regex=r"[0-9]+(\.[0-9]+)?")


@guidance(stateless=True)
def _STRING(lm):
    return lm + gen(regex=r"'([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"")


# ==========================================
# Expressions (explicit depth + length limits)
# ==========================================

@guidance(stateless=True)
def _list_items_inner(lm, level: int):
    # list: expr (',' expr){0..k}
    return lm + _expr(level=level) + _rep_bin("," + optional(_WS) + _expr(level=level), max_rep=_MAX_CALL_ARGS - 1)


@guidance(stateless=True)
def _dict_items_inner(lm, level: int):
    # dict: expr ':' expr (',' expr ':' expr){0..k}
    item = _expr(level=level) + _COLON + _expr(level=level)
    return lm + item + _rep_bin("," + optional(_WS) + item, max_rep=_MAX_CALL_ARGS - 1)


@guidance(stateless=True)
def _arg_list_inner(lm, level: int):
    return lm + _expr(level=level) + _rep_bin("," + optional(_WS) + _expr(level=level), max_rep=_MAX_CALL_ARGS - 1)


@guidance(stateless=True)
def _atom(lm, level: int):
    # Depth cutoff: forbid recursive structures at max depth.
    if level >= _MAX_EXPR_DEPTH:
        return lm + select([_NAME(), _NUMBER(), _STRING(), "None", "True", "False"])

    return lm + select(
        [
            _NAME(),
            _NUMBER(),
            _STRING(),
            "None",
            "True",
            "False",
            "(" + optional(_WS) + _expr(level=level + 1) + optional(_WS) + ")",
            "[" + optional(_list_items_inner(level=level + 1)) + "]",
            "{" + optional(_dict_items_inner(level=level + 1)) + "}",
        ]
    )


@guidance(stateless=True)
def _trailer(lm, level: int):
    # Limit the amount of attribute/call/index chaining
    return lm + select(
        [
            "(" + optional(_arg_list_inner(level=level)) + ")",
            "." + _NAME(),
            "[" + _expr(level=level) + "]",
        ]
    )


@guidance(stateless=True)
def _atom_expr(lm, level: int):
    return lm + _atom(level=level) + _rep_trailer(level=level, max_rep=_MAX_TRAILERS)


@guidance(stateless=True)
def _rep_trailer(lm, level: int, max_rep: int):
    # bounded repetition: trailer{0..max_rep}
    if max_rep <= 0:
        return lm
    return lm + optional(_trailer(level=level) + _rep_trailer(level=level, max_rep=max_rep - 1))


@guidance(stateless=True)
def _power(lm, level: int):
    return lm + _atom_expr(level=level) + optional(optional(_WS) + "**" + optional(_WS) + _factor(level=level))


@guidance(stateless=True)
def _factor(lm, level: int):
    return lm + select(
        [
            _power(level=level),
            select(["+", "-", "~"]) + optional(_WS) + _factor(level=level),
        ]
    )


@guidance(stateless=True)
def _term(lm, level: int):
    op = select(["*", "/", "%", "//"])
    return lm + _factor(level=level) + _rep_bin(optional(_WS) + op + optional(_WS) + _factor(level=level), max_rep=_MAX_BINOPS)


@guidance(stateless=True)
def _arith_expr(lm, level: int):
    op = select(["+", "-"])
    return lm + _term(level=level) + _rep_bin(optional(_WS) + op + optional(_WS) + _term(level=level), max_rep=_MAX_BINOPS)


@guidance(stateless=True)
def _comp_op(lm):
    return lm + select(["<", ">", "==", ">=", "<=", "!=", "in", "not in", "is", "is not"])


@guidance(stateless=True)
def _comparison(lm, level: int):
    # comparison chains are also bounded
    return lm + _arith_expr(level=level) + _rep_bin(optional(_WS) + _comp_op() + optional(_WS) + _arith_expr(level=level), max_rep=2)


@guidance(stateless=True)
def _not_test(lm, level: int):
    # Must advance depth on recursion to avoid infinite 'not not not ...'
    if level >= _MAX_EXPR_DEPTH:
        return lm + _comparison(level=level)
    return lm + select([_comparison(level=level), "not" + _WS + _not_test(level=level + 1)])


@guidance(stateless=True)
def _and_test(lm, level: int):
    return lm + _not_test(level=level) + _rep_bin(optional(_WS) + "and" + _WS + _not_test(level=level), max_rep=_MAX_BINOPS)


@guidance(stateless=True)
def _or_test(lm, level: int):
    return lm + _and_test(level=level) + _rep_bin(optional(_WS) + "or" + _WS + _and_test(level=level), max_rep=_MAX_BINOPS)


@guidance(stateless=True)
def _expr(lm, level: int = 0):
    return lm + _or_test(level=level)


@guidance(stateless=True)
def _rep_bin(lm, chunk, max_rep: int):
    # bounded repetition helper: chunk{0..max_rep}
    if max_rep <= 0:
        return lm
    return lm + optional(chunk + _rep_bin(chunk, max_rep=max_rep - 1))


# ==========================================
# Statements and Blocks (bounded length)
# ==========================================

_block = None  # Forward declaration


@guidance(stateless=True)
def _simple_stmt(lm):
    target = _NAME()  # simplified target
    assign_stmt = target + optional(_WS) + "=" + optional(_WS) + _expr(level=0)
    return_stmt = "return" + optional(_WS + _expr(level=0))
    return lm + select([assign_stmt, return_stmt, _expr(level=0), "pass", "break", "continue"])


@guidance(stateless=True)
def _if_stmt(lm, level: int):
    return (
        lm
        + "if"
        + _WS
        + _expr(level=0)
        + _COLON
        + _block(level + 1)
        + _rep_bin("elif" + _WS + _expr(level=0) + _COLON + _block(level + 1), max_rep=2)
        + optional("else" + _COLON + _block(level + 1))
    )


@guidance(stateless=True)
def _for_stmt(lm, level: int):
    target = _NAME()
    return lm + "for" + _WS + target + _WS + "in" + _WS + _expr(level=0) + _COLON + _block(level + 1)


@guidance(stateless=True)
def _while_stmt(lm, level: int):
    return lm + "while" + _WS + _expr(level=0) + _COLON + _block(level + 1)


@guidance(stateless=True)
def _stmt(lm, level: int):
    if level > _MAX_BLOCK_DEPTH:
        return lm + _simple_stmt()
    return lm + select([
        _simple_stmt(),
        _if_stmt(level=level),
        _for_stmt(level=level),
        _while_stmt(level=level),
    ])


@guidance(stateless=True)
def _block_impl(lm, level: int, remaining: int = _MAX_STMTS_PER_BLOCK):
    # bounded number of statements per block to guarantee termination
    if level > _MAX_BLOCK_DEPTH:
        # At max depth, only allow simple statements, still bounded
        return lm + _NEWLINE + _INDENTS[_MAX_BLOCK_DEPTH] + _simple_stmt() + _NEWLINE

    # Always emit at least one statement
    out = lm + _NEWLINE + _INDENTS[level] + _stmt(level=level) + _NEWLINE
    if remaining <= 1:
        return out
    return out + optional(_block_more(level=level, remaining=remaining - 1))


@guidance(stateless=True)
def _block_more(lm, level: int, remaining: int):
    if remaining <= 0:
        return lm
    out = lm + _INDENTS[level] + _stmt(level=level) + _NEWLINE
    if remaining == 1:
        return out
    return out + optional(_block_more(level=level, remaining=remaining - 1))


_block = _block_impl


# ==========================================
# Top-level Structures
# ==========================================

@guidance(stateless=True)
def _param_list(lm):
    return lm + optional(_NAME() + _rep_bin("," + optional(_WS) + _NAME(), max_rep=4))


@guidance(stateless=True)
def _dotted_name(lm):
    return lm + _NAME() + _rep_bin("." + _NAME(), max_rep=4)


@guidance(stateless=True)
def _import_stmt(lm):
    targets = select(["*", _NAME() + _rep_bin("," + optional(_WS) + _NAME(), max_rep=4)])
    return lm + select([
        "import" + _WS + _dotted_name(),
        "from" + _WS + _dotted_name() + _WS + "import" + _WS + targets,
    ])


@guidance(stateless=True)
def _funcdef(lm):
    return lm + "def" + _WS + _NAME() + "(" + _param_list() + ")" + _COLON + _block(level=1)


@guidance(stateless=True)
def python_grammar_constrained(lm):
    """Main guidance function to generate a Python file with imports and one function.

    Stabilized by:
    - bounding expr depth
    - bounding binop/chain lengths
    - bounding number of statements per block
    """
    _dbg("enter python_grammar_constrained")
    return (
        lm
        + zero_or_more(_NEWLINE)
        + _rep_bin(_import_stmt() + _NEWLINE, max_rep=4)
        + _funcdef()
        + zero_or_more(_NEWLINE)
    )


# ========================================================================================================
# Penman CFG (converted from constraint_templates/penman/penman.cfg)
# ========================================================================================================

@guidance(stateless=True)
def penman_ws(lm):
    """WS: \" \"*"""
    return lm + gen(regex=r" *")


@guidance(stateless=True)
def penman_var(lm):
    """var: /[a-z][0-9]?/"""
    return lm + gen(regex=r"[a-z][0-9]?")


@guidance(stateless=True)
def penman_concept(lm):
    """concept: /[A-Za-z0-9_\-]+/"""
    return lm + gen(regex=r"[A-Za-z0-9_\-]+")


@guidance(stateless=True)
def penman_role_name(lm):
    """role_name: /[A-Z0-9a-z_\-]+/"""
    return lm + gen(regex=r"[A-Za-z0-9_\-]+")


@guidance(stateless=True)
def penman_number(lm):
    """number: /[0-9]+/"""
    return lm + gen(regex=r"[0-9]+")


@guidance(stateless=True)
def penman_string(lm):
    """string: \"\"\" /[A-Za-z0-9_ \-]+/ \"\"\""""
    return lm + "\"" + gen(regex=r"[A-Za-z0-9_ \-]+") + "\""


@guidance(stateless=True)
def penman_literal(lm):
    """literal: string | number | '-'"""
    return lm + select([penman_string(), penman_number(), "-"])


# --- Forward decls (mutual recursion) ---
penman_node = None


@guidance(stateless=True)
def penman_child(lm):
    """child: node | literal | var | concept"""
    return lm + select([penman_node(), penman_literal(), penman_var(), penman_concept()])


@guidance(stateless=True)
def penman_role_block(lm):
    """role_block: WS? ':' role_name WS child"""
    return (
        lm
        + optional(penman_ws())
        + ":"
        + penman_role_name()
        + penman_ws()
        + penman_child()
    )


@guidance(stateless=True)
def penman_node_impl(lm):
    """node: '(' var WS? '/' WS? concept role_block* ')'"""
    return (
        lm
        + "("
        + penman_var()
        + optional(penman_ws())
        + "/"
        + optional(penman_ws())
        + penman_concept()
        + zero_or_more(penman_role_block())
        + ")"
    )


penman_node = penman_node_impl


@guidance(stateless=True)
def penman_graph(lm):
    """graph: node"""
    return lm + penman_node()


@guidance(stateless=True)
def penman_start(lm, name: str = "penman"):
    """start: graph

    用法：lm += penman_start(name='output')
    然后用 lm['output'] 取出（如果你在外层用 capture/call 也可以）。
    """
    _dbg("enter penman_start")
    return lm + penman_graph()
