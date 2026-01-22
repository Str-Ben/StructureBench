from lark import Lark
from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


grammar = _read_text("./latex/math.cfg")

test_input = _read_text("./gmrchecker_input.txt")
# 预处理：删除所有空格（不删除换行）
test_input = test_input.replace(" ", "")


parser = Lark(grammar, start="start", parser="earley")  # or parser="earley" or parser="lalr"
tree = parser.parse(test_input)
print(tree.pretty())