from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from lumen.errors import ToolError
from lumen.tools.base import Tool

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000


def evaluate(expression: str) -> float:
    if len(expression) > 500:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        return _eval(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ArithmeticError, ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"cannot evaluate expression: {exc}") from exc


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ToolError("booleans are not supported")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError("exponent is too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ToolError(f"unsupported function: {node.func.id}")
        return function(*[_eval(argument) for argument in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression exactly. Supports + - * / // % **, parentheses, "
        "sqrt, log, abs, round, min, max, and the constants pi and e."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The arithmetic expression to evaluate"},
        },
        "required": ["expression"],
    }

    def run(self, arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("expression must be a non-empty string")
        result = evaluate(expression)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        return str(result)
