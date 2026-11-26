"""Hunch language front-end: lexer, parser, AST.

The language is deliberately tiny so that both humans and AI agents can
learn it from a single page. See llms.txt for the user-facing cheat sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


# Source text of the most recent parse, keyed by path, so that an error raised
# deep in the compiler can still show the offending line. Cheap: a program is
# a few kilobytes and there is one per run.
_SOURCES: dict[str, str] = {}


class HunchError(Exception):
    """A user-facing compile/run error with a source location.

    The rendered form is what the CLI prints, so every error in the toolchain
    gets a location, the offending line, a caret under the token and, where the
    compiler knows one, a suggested fix.
    """

    def __init__(self, message: str, line: int = 0, col: int = 0, path: str = "",
                 hint: str = ""):
        self.message = message
        self.line = line
        self.col = col
        self.path = path
        self.hint = hint
        super().__init__(self.render())

    def source_line(self) -> Optional[str]:
        src = _SOURCES.get(self.path, "")
        if not src or self.line <= 0:
            return None
        lines = src.splitlines()
        return lines[self.line - 1] if self.line <= len(lines) else None

    def render(self) -> str:
        if self.path and self.line > 0:
            loc = f"{self.path}:{self.line}:{self.col}"
        elif self.path:
            loc = f"{self.path}"
        elif self.line > 0:
            loc = f"line {self.line}"
        else:
            loc = ""
        head = f"{loc}: error: {self.message}" if loc else f"error: {self.message}"
        out = [head]
        text = self.source_line()
        if text is not None:
            gutter = " " * len(str(self.line))
            out.append(f"{gutter} |")
            out.append(f"{self.line} | {text}")
            if 0 < self.col <= len(text) + 1:
                width = _token_width(text, self.col)
                out.append(f"{gutter} | {' ' * (self.col - 1)}{'^' * width}")
            out.append(f"{gutter} |")
        if self.hint:
            out.append(f"{' ' * len(str(self.line))} = help: {self.hint}")
        return "\n".join(out)


def _token_width(text: str, col: int) -> int:
    """Width of the token starting at 1-based `col`, so the caret fits it."""
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S", text[col - 1:])
    return len(match.group(0)) if match else 1


def nearest(name: str, candidates) -> Optional[str]:
    """Closest candidate name, if one is close enough to be a likely typo."""
    if len(name) < 3:
        return None
    best, best_score = None, 3
    for cand in candidates:
        # A typo keeps the first letter and roughly the shape. Without this,
        # short names produce absurd suggestions ("y" -> "a").
        if not cand or cand[0].lower() != name[0].lower():
            continue
        if abs(len(cand) - len(name)) > 2:
            continue
        score = _edit_distance(name.lower(), cand.lower())
        if score < best_score:
            best, best_score = cand, score
    return best if best_score <= 2 else None


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

KEYWORDS = {
    "world", "horizon", "state", "init", "step", "fn",
    "let", "if", "else", "while", "for", "in", "return",
    "hunch", "reward", "done", "assert", "param", "record", "true", "false",
}

# Longest-first so that '..' wins over '.', '==' over '=', etc.
OPERATORS = [
    "&&", "||", "==", "!=", "<=", ">=", "..",
    "+", "-", "*", "/", "%", "<", ">", "!", "=",
    "(", ")", "{", "}", "[", "]", ",", ";", ":", ".",
]


@dataclass
class Token:
    kind: str      # IDENT, NUM, KW, OP, EOF
    value: str
    line: int
    col: int
    pos: int = 0   # offset of the token in the source, and of its end
    end: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind}, {self.value!r}, {self.line}:{self.col})"


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(source: str, path: str = "") -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    line_start = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # Whitespace
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            line_start = i
            continue

        # Comments
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue

        col = i - line_start + 1

        if source.startswith("/*", i):
            raise HunchError("there are no /* */ comments", line, col, path,
                             "a comment runs from // to the end of the line")

        # Numbers
        m = _NUM_RE.match(source, i)
        if m:
            text = m.group(0)
            # Guard against '1..5' being read as a float.
            end = m.end()
            if end < n and source[end] == "." and not text.endswith("."):
                if end + 1 < n and source[end + 1] == ".":
                    pass  # leave the '..' for the operator scanner
            tokens.append(Token("NUM", text, line, col, i, end))
            i = end
            continue

        # Identifiers / keywords
        m = _IDENT_RE.match(source, i)
        if m:
            text = m.group(0)
            kind = "KW" if text in KEYWORDS else "IDENT"
            tokens.append(Token(kind, text, line, col, i, m.end()))
            i = m.end()
            continue

        # Operators / punctuation
        for op in OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token("OP", op, line, col, i, i + len(op)))
                i += len(op)
                break
        else:
            raise HunchError(f"unexpected character {ch!r}", line, col, path)

    tokens.append(Token("EOF", "", line, i - line_start + 1, i, i))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Node:
    line: int = 0
    col: int = 0


# --- expressions -----------------------------------------------------------

@dataclass
class Num(Node):
    value: float = 0.0
    is_int: bool = True


@dataclass
class Name(Node):
    name: str = ""


@dataclass
class Index(Node):
    base: str = ""
    index: Node = None  # type: ignore[assignment]


@dataclass
class Call(Node):
    name: str = ""
    args: List[Node] = field(default_factory=list)


@dataclass
class BinOp(Node):
    op: str = ""
    left: Node = None  # type: ignore[assignment]
    right: Node = None  # type: ignore[assignment]


@dataclass
class UnOp(Node):
    op: str = ""
    operand: Node = None  # type: ignore[assignment]


@dataclass
class ArrayLit(Node):
    items: List[Node] = field(default_factory=list)


@dataclass
class Hunch(Node):
    agent: str = ""
    obs: Any = None          # List[Node] for a literal, or str for a state array
    obs_expr: Optional[Node] = None  # single expression, folded into a 1-vector
    n_actions: int = 2
    legal: Optional[Node] = None     # legal-move mask: a Name (an array) or an ArrayLit
    action_names: Optional[List[str]] = None   # hunch p(obs, [left, right])
    obs_labels: List[str] = field(default_factory=list)   # source text of each value


# --- statements ------------------------------------------------------------

@dataclass
class Let(Node):
    name: str = ""
    type_name: Optional[str] = None
    array_len: int = 0
    value: Node = None  # type: ignore[assignment]


@dataclass
class Assign(Node):
    target: Any = None   # str or Index
    value: Node = None   # type: ignore[assignment]


@dataclass
class If(Node):
    cond: Node = None  # type: ignore[assignment]
    then_body: List[Node] = field(default_factory=list)
    else_body: List[Node] = field(default_factory=list)


@dataclass
class While(Node):
    cond: Node = None  # type: ignore[assignment]
    body: List[Node] = field(default_factory=list)


@dataclass
class For(Node):
    var: str = ""
    start: Node = None  # type: ignore[assignment]
    end: Node = None    # type: ignore[assignment]
    body: List[Node] = field(default_factory=list)


@dataclass
class Return(Node):
    value: Optional[Node] = None


@dataclass
class ExprStmt(Node):
    expr: Node = None  # type: ignore[assignment]


@dataclass
class Reward(Node):
    agent: Optional[str] = None
    value: Node = None  # type: ignore[assignment]
    agent_at: Optional[Node] = None  # where the agent name is, for diagnostics


@dataclass
class RecordDecl(Node):
    name: str = ""
    fields: List[tuple] = field(default_factory=list)   # (field name, type name)


@dataclass
class Field(Node):
    base: str = ""
    index: Optional[Node] = None
    field: str = ""


@dataclass
class Assert(Node):
    cond: Node = None


@dataclass
class DoneStmt(Node):
    pass


@dataclass
class VarDecl(Node):
    name: str = ""
    type_name: str = "int"
    array_len: int = 0          # 0 means scalar
    default: Optional[Node] = None


@dataclass
class FuncDecl(Node):
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class Program:
    world: str = "world"
    horizon: int = 100
    vars: List[VarDecl] = field(default_factory=list)
    init: List[Node] = field(default_factory=list)
    step: List[Node] = field(default_factory=list)
    funcs: List[FuncDecl] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    path: str = ""
    # Named actions: name -> (index, where it was first given). A name is a
    # constant everywhere in the world, so it has one index.
    action_names: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

TYPE_NAMES = {"int", "float", "bool", "num"}


class Parser:
    records: dict = {}

    def __init__(self, tokens: List[Token], path: str = ""):
        self.toks = tokens
        self.pos = 0
        self.path = path
        self.action_names: dict = {}

    # -- helpers ----------------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.toks[self.pos]

    def peek(self, off: int = 1) -> Token:
        idx = min(self.pos + off, len(self.toks) - 1)
        return self.toks[idx]

    def error(self, msg: str, tok: Optional[Token] = None, hint: str = "") -> HunchError:
        t = tok or self.cur
        return HunchError(msg, t.line, t.col, self.path, hint)

    def at(self, kind: str, value: Optional[str] = None) -> bool:
        t = self.cur
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def accept(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        if self.at(kind, value):
            t = self.cur
            self.pos += 1
            return t
        return None

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.accept(kind, value)
        if t is None:
            want = value if value is not None else kind
            got = self.cur.value or "end of file"
            raise self.error(f"expected {want!r}, got {got!r}")
        return t

    def expect_ident(self) -> str:
        return self.expect("IDENT").value

    # -- top level --------------------------------------------------------
    def parse_program(self) -> Program:
        prog = Program(path=self.path)
        self.records = prog.records
        self.action_names = prog.action_names

        if self.accept("KW", "world"):
            prog.world = self.expect_ident()

        while not self.at("EOF"):
            if self.accept("KW", "horizon"):
                tok = self.expect("NUM")
                prog.horizon = int(float(tok.value))
                if prog.horizon <= 0:
                    raise self.error("horizon must be positive", tok)
                self.accept("OP", ";")
            elif self.accept("KW", "param"):
                # param NAME = value;  -- a compile-time constant that can be
                # overridden from the command line without editing the world.
                start_tok = self.cur
                name = self.expect_ident()
                self.expect("OP", "=")
                value_tok = self.cur
                if not (value_tok.kind == "NUM" or
                        (value_tok.kind == "OP" and value_tok.value == "-")):
                    raise self.error("a param needs a number", value_tok)
                sign = 1.0
                if self.accept("OP", "-"):
                    sign = -1.0
                    value_tok = self.cur
                value = self.expect("NUM")
                self.accept("OP", ";")
                if name in prog.params:
                    raise self.error(f"param '{name}' already declared", start_tok)
                prog.params[name] = sign * float(value.value)
            elif self.at("KW", "record"):
                self.parse_record(prog)
            elif self.at("KW", "state"):
                self.parse_state(prog)
            elif self.at("KW", "init"):
                self.advance()
                prog.init = self.parse_block()
            elif self.at("KW", "step"):
                self.advance()
                prog.step = self.parse_block()
            elif self.at("KW", "fn"):
                prog.funcs.append(self.parse_fn())
            else:
                raise self.error(f"unexpected {self.cur.value!r} at top level")

        if not prog.step:
            raise self.error("program has no 'step' block")
        # A state block is optional: a world with nothing to remember should
        # not have to declare an empty one.
        return prog

    def advance(self) -> Token:
        t = self.cur
        self.pos += 1
        return t

    def parse_record(self, prog: Program) -> None:
        tok = self.expect("KW", "record")
        name = self.expect_ident()
        if name in prog.records:
            raise self.error(f"record '{name}' is declared twice", tok)
        self.expect("OP", "{")
        fields = []
        while not self.accept("OP", "}"):
            if self.at("EOF"):
                raise self.error("unterminated 'record' block")
            fname = self.expect_ident()
            self.expect("OP", ":")
            ftype = self.cur
            if ftype.value not in TYPE_NAMES:
                raise self.error(
                    f"unknown field type {ftype.value!r}; records hold int, float, bool or num")
            self.advance()
            self.accept("OP", ";")
            if fname in (f for f, _ in fields):
                raise self.error(f"field '{fname}' is declared twice", ftype)
            fields.append((fname, ftype.value))
        if not fields:
            raise self.error(f"record '{name}' has no fields", tok)
        prog.records[name] = RecordDecl(line=tok.line, col=tok.col, name=name,
                                        fields=fields)

    def parse_state(self, prog: Program) -> None:
        self.expect("KW", "state")
        self.expect("OP", "{")
        while not self.accept("OP", "}"):
            if self.at("EOF"):
                raise self.error("unterminated 'state' block")
            prog.vars.append(self.parse_var_decl())

    def parse_var_decl(self) -> VarDecl:
        tok = self.expect("IDENT")
        self.expect("OP", ":")
        type_tok = self.cur
        known = type_tok.kind in ("IDENT", "KW") and (
            type_tok.value in TYPE_NAMES or type_tok.value in self.records)
        if not known:
            raise self.error(
                f"unknown type {type_tok.value!r}; use int, float, bool, num, or a record "
                "declared above 'state'", type_tok)
        self.advance()
        array_len = 0
        if self.accept("OP", "["):
            n = self.expect("NUM")
            array_len = int(float(n.value))
            if array_len <= 0:
                raise self.error("array length must be positive")
            self.expect("OP", "]")
        default = None
        if self.accept("OP", "="):
            default = self.parse_expr()
        self.expect("OP", ";")
        return VarDecl(line=tok.line, col=tok.col, name=tok.value, type_name=type_tok.value,
                       array_len=array_len, default=default)

    def parse_fn(self) -> FuncDecl:
        tok = self.expect("KW", "fn")
        name = self.expect_ident()
        self.expect("OP", "(")
        params: List[str] = []
        if not self.at("OP", ")"):
            while True:
                params.append(self.expect_ident())
                if not self.accept("OP", ","):
                    break
        self.expect("OP", ")")
        body = self.parse_block()
        return FuncDecl(line=tok.line, col=tok.col, name=name, params=params, body=body)

    def parse_block(self) -> List[Node]:
        self.expect("OP", "{")
        body: List[Node] = []
        while not self.accept("OP", "}"):
            if self.at("EOF"):
                raise self.error("unterminated block")
            body.append(self.parse_stmt())
        return body

    # -- statements -------------------------------------------------------
    def parse_stmt(self) -> Node:
        tok = self.cur

        if tok.kind == "IDENT" and tok.value in ("elif", "elsif", "elseif"):
            raise self.error(f"there is no '{tok.value}'", hint="write `else if`")

        if self.at("KW", "let"):
            self.advance()
            name = self.expect_ident()
            type_name = None
            if self.accept("OP", ":"):
                t = self.cur
                if t.value not in TYPE_NAMES:
                    raise self.error(f"unknown type {t.value!r}")
                self.advance()
                type_name = t.value
            array_len = 0
            if self.accept("OP", "["):
                n = self.expect("NUM")
                array_len = int(float(n.value))
                if array_len <= 0:
                    raise self.error("array length must be positive")
                self.expect("OP", "]")
            val = None
            if self.accept("OP", "="):
                val = self.parse_expr()
            self.expect("OP", ";")
            return Let(line=tok.line, col=tok.col, name=name, type_name=type_name,
                       array_len=array_len, value=val)

        if self.at("KW", "assert"):
            self.advance()
            cond = self.parse_expr()
            self.expect("OP", ";")
            return Assert(line=tok.line, col=tok.col, cond=cond)

        if self.at("KW", "if"):
            self.advance()
            cond = self.parse_expr()
            then_body = self.parse_block()
            else_body: List[Node] = []
            if self.accept("KW", "else"):
                if self.at("KW", "if"):
                    else_body = [self.parse_stmt()]
                else:
                    else_body = self.parse_block()
            return If(line=tok.line, col=tok.col, cond=cond, then_body=then_body, else_body=else_body)

        if self.at("KW", "while"):
            self.advance()
            cond = self.parse_expr()
            body = self.parse_block()
            return While(line=tok.line, col=tok.col, cond=cond, body=body)

        if self.at("KW", "for"):
            self.advance()
            var = self.expect_ident()
            self.expect("KW", "in")
            start = self.parse_expr()
            self.expect("OP", "..")
            end = self.parse_expr()
            body = self.parse_block()
            return For(line=tok.line, col=tok.col, var=var, start=start, end=end, body=body)

        if self.at("KW", "return"):
            self.advance()
            val = None
            if not self.at("OP", ";"):
                val = self.parse_expr()
            self.expect("OP", ";")
            return Return(line=tok.line, col=tok.col, value=val)

        if self.at("KW", "done"):
            self.advance()
            self.expect("OP", "(")
            self.expect("OP", ")")
            self.expect("OP", ";")
            return DoneStmt(line=tok.line, col=tok.col)

        if self.at("KW", "reward"):
            self.advance()
            self.expect("OP", "(")
            agent: Optional[str] = None
            agent_at: Optional[Node] = None
            # reward(alice, expr) vs reward(expr)
            if self.cur.kind == "IDENT" and self.peek().kind == "OP" and self.peek().value == ",":
                name_tok = self.advance()
                agent = name_tok.value
                agent_at = Name(line=name_tok.line, col=name_tok.col, name=agent)
                self.expect("OP", ",")
            val = self.parse_expr()
            self.expect("OP", ")")
            self.expect("OP", ";")
            return Reward(line=tok.line, col=tok.col, agent=agent, value=val,
                          agent_at=agent_at)

        # assignment or expression statement: parse a target-shaped prefix,
        # then decide. Works for name, name[i], name.field and name[i].field.
        if self.cur.kind == "IDENT":
            save = self.pos
            target = None
            try:
                target = self.parse_postfix()
            except HunchError:
                target = None
            if (target is not None and isinstance(target, (Name, Index, Field))
                    and self.at("OP", "=")):
                self.advance()
                val = self.parse_expr()
                self.expect("OP", ";")
                return Assign(line=tok.line, col=tok.col, target=target, value=val)
            self.pos = save

        expr = self.parse_expr()
        self.expect("OP", ";")
        return ExprStmt(line=tok.line, col=tok.col, expr=expr)

    # -- expressions ------------------------------------------------------
    def parse_expr(self) -> Node:
        return self.parse_or()

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.at("OP", "||"):
            tok = self.advance()
            rhs = self.parse_and()
            node = BinOp(line=tok.line, col=tok.col, op="||", left=node, right=rhs)
        return node

    def parse_and(self) -> Node:
        node = self.parse_cmp()
        while self.at("OP", "&&"):
            tok = self.advance()
            rhs = self.parse_cmp()
            node = BinOp(line=tok.line, col=tok.col, op="&&", left=node, right=rhs)
        return node

    def parse_cmp(self) -> Node:
        node = self.parse_add()
        while self.cur.kind == "OP" and self.cur.value in ("==", "!=", "<", "<=", ">", ">="):
            tok = self.advance()
            rhs = self.parse_add()
            node = BinOp(line=tok.line, col=tok.col, op=tok.value, left=node, right=rhs)
        return node

    def parse_add(self) -> Node:
        node = self.parse_mul()
        while self.cur.kind == "OP" and self.cur.value in ("+", "-"):
            tok = self.advance()
            rhs = self.parse_mul()
            node = BinOp(line=tok.line, col=tok.col, op=tok.value, left=node, right=rhs)
        return node

    def parse_mul(self) -> Node:
        node = self.parse_unary()
        while self.cur.kind == "OP" and self.cur.value in ("*", "/", "%"):
            tok = self.advance()
            rhs = self.parse_unary()
            node = BinOp(line=tok.line, col=tok.col, op=tok.value, left=node, right=rhs)
        return node

    def parse_unary(self) -> Node:
        if self.cur.kind == "OP" and self.cur.value in ("-", "!"):
            tok = self.advance()
            operand = self.parse_unary()
            return UnOp(line=tok.line, col=tok.col, op=tok.value, operand=operand)
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        while True:
            if self.at("OP", "["):
                tok = self.advance()
                idx = self.parse_expr()
                self.expect("OP", "]")
                if not isinstance(node, Name):
                    raise self.error("only a variable can be indexed", tok)
                node = Index(line=tok.line, col=tok.col, base=node.name, index=idx)
                continue
            if self.at("OP", "."):
                dot = self.advance()
                field = self.expect_ident()
                if isinstance(node, Index):
                    node = Field(line=dot.line, col=dot.col, base=node.base,
                                 index=node.index, field=field)
                elif isinstance(node, Name):
                    node = Field(line=dot.line, col=dot.col, base=node.name,
                                 index=None, field=field)
                else:
                    raise self.error("only a variable can have a field", dot)
                continue
            return node

    def parse_primary(self) -> Node:
        tok = self.cur

        if tok.kind == "NUM":
            self.advance()
            is_int = "." not in tok.value
            return Num(line=tok.line, col=tok.col, value=float(tok.value), is_int=is_int)

        if tok.kind == "KW" and tok.value in ("true", "false"):
            self.advance()
            return Num(line=tok.line, col=tok.col, value=1.0 if tok.value == "true" else 0.0, is_int=True)

        if tok.kind == "KW" and tok.value == "hunch":
            return self.parse_hunch()

        if tok.kind == "OP" and tok.value == "(":
            self.advance()
            node = self.parse_expr()
            self.expect("OP", ")")
            return node

        if tok.kind == "OP" and tok.value == "[":
            self.advance()
            items: List[Node] = []
            if not self.at("OP", "]"):
                while True:
                    items.append(self.parse_expr())
                    if not self.accept("OP", ","):
                        break
            self.expect("OP", "]")
            return ArrayLit(line=tok.line, col=tok.col, items=items)

        if tok.kind == "IDENT":
            self.advance()
            if self.at("OP", "("):
                self.advance()
                args: List[Node] = []
                if not self.at("OP", ")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.accept("OP", ","):
                            break
                self.expect("OP", ")")
                return Call(line=tok.line, col=tok.col, name=tok.value, args=args)
            return Name(line=tok.line, col=tok.col, name=tok.value)

        raise self.error(f"unexpected token {tok.value or 'end of file'!r}")

    def parse_hunch(self) -> Hunch:
        tok = self.expect("KW", "hunch")
        agent = self.expect_ident()
        self.expect("OP", "(")
        labels: List[str] = []
        if self.at("OP", "["):
            self.advance()
            items: List[Node] = []
            if not self.at("OP", "]"):
                while True:
                    start = self.pos
                    items.append(self.parse_expr())
                    labels.append(self.text_between(start, self.pos))
                    if not self.accept("OP", ","):
                        break
            self.expect("OP", "]")
            obs: Any = items
            obs_expr = None
        else:
            start = self.pos
            expr = self.parse_expr()
            if isinstance(expr, Name):
                obs = expr.name
                obs_expr = None
            else:
                obs = None
                obs_expr = expr
                labels.append(self.text_between(start, self.pos))
        self.expect("OP", ",")
        # The actions: how many, or their names. A name is a constant with the
        # value of its position, usable anywhere in the world.
        action_names: Optional[List[str]] = None
        if self.at("OP", "["):
            self.advance()
            action_names = []
            while True:
                name_tok = self.cur
                if name_tok.kind != "IDENT":
                    raise self.error(
                        "an action list holds names, e.g. [left, right]", name_tok,
                        f"or give the number of actions: hunch {agent}(obs, 2)")
                self.advance()
                if name_tok.value in action_names:
                    raise self.error(f"action '{name_tok.value}' is listed twice", name_tok)
                index = len(action_names)
                prev = self.action_names.get(name_tok.value)
                if prev is not None and prev[0] != index:
                    raise self.error(
                        f"action '{name_tok.value}' is number {index} here and number "
                        f"{prev[0]} at line {prev[1]}", name_tok,
                        "a name stands for one number everywhere; keep the order the "
                        "same or rename one")
                if prev is None:
                    self.action_names[name_tok.value] = (index, name_tok.line, name_tok.col)
                action_names.append(name_tok.value)
                if not self.accept("OP", ","):
                    break
            self.expect("OP", "]")
            n_actions = len(action_names)
        else:
            n_tok = self.expect("NUM")
            n_actions = int(float(n_tok.value))
        # Optional legal-move mask: an array by name, or a [...] list. Anything
        # else (a scalar expression, an indexed element) cannot say which of
        # the n actions are allowed, so it is refused here with the reason.
        legal: Optional[Node] = None
        if self.accept("OP", ","):
            m_tok = self.cur
            if self.at("OP", "["):
                legal = self.parse_primary()
            elif m_tok.kind == "IDENT" and self.peek().kind == "OP" and self.peek().value == ")":
                self.advance()
                legal = Name(line=m_tok.line, col=m_tok.col, name=m_tok.value)
            else:
                raise HunchError(
                    "a legal-move mask must be an array name or a [...] list",
                    m_tok.line, m_tok.col, self.path,
                    f"one value per action, e.g. hunch {agent}(obs, {n_actions}, ok) "
                    f"with 'let ok: int[{n_actions}];'")
        self.expect("OP", ")")
        return Hunch(line=tok.line, col=tok.col, agent=agent, obs=obs, obs_expr=obs_expr,
                     n_actions=n_actions, legal=legal, action_names=action_names,
                     obs_labels=labels)

    def text_between(self, start: int, end: int) -> str:
        """The source text of tokens start..end-1, on one line."""
        if end <= start:
            return ""
        src = _SOURCES.get(self.path, "")
        text = src[self.toks[start].pos:self.toks[end - 1].end]
        return " ".join(text.split())


def parse(source: str, path: str = "") -> Program:
    _SOURCES[path] = source
    tokens = tokenize(source, path)
    return Parser(tokens, path).parse_program()
