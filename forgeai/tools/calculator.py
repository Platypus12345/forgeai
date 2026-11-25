"""Exact arithmetic, evaluated with an AST allow-list rather than eval()."""

from __future__ import annotations

import ast
import math
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("division by zero")
        if type(node.op) is ast.Pow and abs(right) > 64:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function not allowed: {node.func.id}")
        return fn(*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def run(expression: str = "", **_ignored) -> str:
    expression = (expression or "").strip().rstrip("=").strip()
    if not expression:
        return "ERROR: no expression given"
    if len(expression) > 300:
        return "ERROR: expression too long"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"ERROR: could not parse '{expression}': {exc.msg}"
    try:
        value = _eval(tree)
    except Exception as exc:
        return f"ERROR: {exc}"
    if isinstance(value, float):
        rounded = round(value, 6)
        text = f"{int(rounded)}" if rounded == int(rounded) else f"{rounded}"
    else:
        text = str(value)
    return f"{expression} = {text}"
