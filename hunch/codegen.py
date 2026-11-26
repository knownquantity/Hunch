"""Hunch bytecode compiler.

Emits a compact `.hbc` binary that the C virtual machine (`runtime/hunch_vm.c`)
loads and executes. The bytecode is deliberately simple: a stack machine over
doubles, with a flat state array so that the entire universe (values + PRNG)
can be snapshotted with a single memcpy.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from .lang import (
    ArrayLit, Assign, BinOp, Call, DoneStmt, ExprStmt, For, FuncDecl, Hunch,
    HunchError, Assert, Field, If, Index, Let, Name, Node, Num, Program, Return,
    Reward, nearest,
    UnOp, While,
)

# ---------------------------------------------------------------------------
# Opcodes -- must stay in sync with runtime/hunch_vm.c
# ---------------------------------------------------------------------------

HALT = 0
PUSH_CONST = 1
PUSH_VAR = 2
STORE_VAR = 3
PUSH_IDX = 4
STORE_IDX = 5
POP = 6
DUP = 7
ADD = 8
SUB = 9
MUL = 10
DIV = 11
MOD = 12
NEG = 13
EQ = 14
NE = 15
LT = 16
LE = 17
GT = 18
GE = 19
AND = 20
OR = 21
NOT = 22
JUMP = 23
JUMP_FALSE = 24
CALL = 25
RET = 26
SCRATCH = 27
COPY_VARS = 28
HUNCH = 29
REWARD = 30
DONE = 31
RAND = 32
RANDINT = 33
ABS = 34
MIN = 35
MAX = 36
CLAMP = 37
FLOOR = 38
SQRT = 39
EXP = 40
STEP_COUNT = 41
ASSERT = 42
LEGAL = 43
ZERO = 44

OP_NAMES = {
    HALT: "HALT", PUSH_CONST: "PUSH_CONST", PUSH_VAR: "PUSH_VAR",
    STORE_VAR: "STORE_VAR", PUSH_IDX: "PUSH_IDX", STORE_IDX: "STORE_IDX",
    POP: "POP", DUP: "DUP", ADD: "ADD", SUB: "SUB", MUL: "MUL", DIV: "DIV",
    MOD: "MOD", NEG: "NEG", EQ: "EQ", NE: "NE", LT: "LT", LE: "LE", GT: "GT",
    GE: "GE", AND: "AND", OR: "OR", NOT: "NOT", JUMP: "JUMP",
    JUMP_FALSE: "JUMP_FALSE", CALL: "CALL", RET: "RET", SCRATCH: "SCRATCH",
    COPY_VARS: "COPY_VARS", HUNCH: "HUNCH", REWARD: "REWARD", DONE: "DONE",
    RAND: "RAND", RANDINT: "RANDINT", ABS: "ABS", MIN: "MIN", MAX: "MAX",
    CLAMP: "CLAMP", FLOOR: "FLOOR", SQRT: "SQRT", EXP: "EXP",
    ASSERT: "ASSERT",
    STEP_COUNT: "STEP_COUNT", LEGAL: "LEGAL", ZERO: "ZERO",
}

BIN_OPS = {
    "+": ADD, "-": SUB, "*": MUL, "/": DIV, "%": MOD,
    "==": EQ, "!=": NE, "<": LT, "<=": LE, ">": GT, ">=": GE,
    "&&": AND, "||": OR,
}

BUILTIN_ARITY = {
    "rand": 0, "step_count": 0,
    "randint": 2, "abs": 1, "min": 2, "max": 2,
    "clamp": 3, "floor": 1, "sqrt": 1, "exp": 1,
}

BUILTIN_OP = {
    "rand": RAND, "step_count": STEP_COUNT,
    "randint": RANDINT, "abs": ABS, "min": MIN, "max": MAX,
    "clamp": CLAMP, "floor": FLOOR, "sqrt": SQRT, "exp": EXP,
}

MAX_SLOTS = 512
MAX_OBS = 64
MAX_ACTIONS = 32
MAX_AGENTS = 8


class Insn:
    __slots__ = ("op", "a", "b", "c")

    def __init__(self, op: int, a: int = 0, b: int = 0, c: int = 0):
        self.op = op
        self.a = a
        self.b = b
        self.c = c


def _literal(node) -> Optional[float]:
    """The value of a number literal, with an optional minus sign; None for anything else."""
    if isinstance(node, Num):
        return node.value
    if isinstance(node, UnOp) and node.op == "-" and isinstance(node.operand, Num):
        return -node.operand.value
    return None


class CodeGen:
    def __init__(self, prog: Program):
        self.prog = prog
        self.code: List[Insn] = []
        self.consts: List[float] = []
        self._const_map: Dict[float, int] = {}

        self.nslots = 0
        self.var_slots: Dict[str, Tuple[int, int]] = {}
        self.decls: List[Tuple[str, int, int, int]] = []

        self.agents: List[str] = []
        self.agent_idx: Dict[str, int] = {}
        self.agent_obs: Dict[str, int] = {}
        self.agent_acts: Dict[str, int] = {}
        # What `hunch run`, `why` and `play` print instead of bare numbers.
        self.agent_action_names: Dict[str, List[str]] = {}
        self.agent_obs_labels: Dict[str, List[str]] = {}

        self.func_index: Dict[str, int] = {
            fn.name: i for i, fn in enumerate(prog.funcs)
        }
        self.func_info: Dict[str, dict] = {}
        self.call_graph: Dict[str, set] = {fn.name: set() for fn in prog.funcs}
        self._current_fn: Optional[str] = None

        self.scopes: List[Dict[str, int]] = [{}]
        self.bare_rewards: List[int] = []
        self.named_rewards: List[Reward] = []

        self.n_asserts = 0
        self.var_records: Dict[str, Tuple[int, int, int, str]] = {}
        self.init_entry = 0
        self.step_entry = 0
        self.horizon = max(1, min(prog.horizon, 100000))

    # -- helpers ----------------------------------------------------------
    def err(self, msg: str, node: Node = None, hint: str = "") -> HunchError:
        return HunchError(msg, node.line if node is not None else 0,
                          getattr(node, "col", 0) if node is not None else 0,
                          self.prog.path, hint)

    def alloc(self, n: int = 1, node: Node = None) -> int:
        if self.nslots + n > MAX_SLOTS:
            raise self.err(
                f"program needs more than {MAX_SLOTS} state slots; "
                "reduce variables or locals", node)
        base = self.nslots
        self.nslots += n
        return base

    def push_scope(self) -> None:
        self.scopes.append({})

    def pop_scope(self) -> None:
        self.scopes.pop()

    def lookup(self, name: str) -> Optional[int]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def not_an_action(self, name: str, node: Node = None, what: str = "variable") -> None:
        got = self.prog.action_names.get(name)
        if got is not None:
            raise self.err(
                f"'{name}' is the name of an action (line {got[1]}), so it cannot also "
                f"be a {what}", node, "rename one of them")

    def declare_array(self, name: str, n: int, node: Node = None) -> int:
        self.not_an_action(name, node)
        if self.lookup(name) is not None:
            raise self.err(f"'{name}' is already declared", node)
        base = self.alloc(n, node)
        self.scopes[-1][name] = base
        self.var_slots[name] = (base, n)
        return base

    def declare(self, name: str, node: Node = None) -> int:
        self.not_an_action(name, node)
        if self.lookup(name) is not None:
            raise self.err(f"'{name}' is already declared", node)
        slot = self.alloc(1, node)
        self.scopes[-1][name] = slot
        return slot

    def emit(self, op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
        self.code.append(Insn(op, a, b, c))
        return len(self.code) - 1

    @property
    def here(self) -> int:
        return len(self.code)

    def const(self, value: float) -> int:
        key = float(value)
        if key in self._const_map:
            return self._const_map[key]
        idx = len(self.consts)
        self.consts.append(key)
        self._const_map[key] = idx
        return idx

    def patch(self, at: int, target: int) -> None:
        self.code[at].a = target

    @staticmethod
    def pos(node: Node) -> int:
        """A source position packed into one operand: line * 4096 + column.

        PUSH_IDX and STORE_IDX have only their third operand free, and
        `hunch test` needs to say where an index went out of range.
        """
        return node.line * 4096 + min(node.col, 4095)

    # -- top level --------------------------------------------------------
    def compile(self) -> dict:
        for name in self.prog.params:
            self.not_an_action(name, None, "param")
        for fn in self.prog.funcs:
            self.not_an_action(fn.name, fn, "function")
        for v in self.prog.vars:
            self.not_an_action(v.name, v, "state variable")

        # State variables occupy slots 0..n-1 in declaration order.
        defaults = self._declare_state()

        # Functions are compiled first so their frames sit at the bottom of
        # the slot map; CALL refers to a function *index*, so forward
        # references are fine.
        seen = {}
        for fn in self.prog.funcs:
            if fn.name in seen:
                raise self.err(f"function '{fn.name}' is defined twice", fn,
                               "give one of them a different name")
            if fn.name in self.var_slots:
                raise self.err(
                    f"function '{fn.name}' has the same name as a state variable",
                    fn, "rename one of them")
            seen[fn.name] = fn
            self.compile_fn(fn)

        self.init_entry = self.here
        self.push_scope()
        for stmt in self.prog.init:
            self.stmt(stmt)
        self.pop_scope()
        self.emit(HALT)

        self.step_entry = self.here
        self.push_scope()
        for stmt in self.prog.step:
            self.stmt(stmt)
        self.pop_scope()
        self.emit(HALT)
        self.emit(HALT)

        self.check_recursion()
        self.check_reward_agents()
        self.resolve_bare_rewards()

        defaults = defaults + [0.0] * (self.nslots - len(defaults))

        return {
            "world": self.prog.world,
            "horizon": self.horizon,
            "nslots": self.nslots,
            "n_asserts": self.n_asserts,
            "defaults": defaults,
            "consts": self.consts,
            "code": self.code,
            "agents": self.agents,
            "agent_obs": self.agent_obs,
            "agent_acts": self.agent_acts,
            "agent_action_names": self.agent_action_names,
            "agent_obs_labels": self.agent_obs_labels,
            "decls": self.decls,
            "funcs": {name: self.func_info[name] for name in self.func_order()},
            "init_entry": self.init_entry,
            "step_entry": self.step_entry,
            "path": self.prog.path,
        }

    def func_order(self) -> List[str]:
        return [fn.name for fn in self.prog.funcs]

    def _declare_state(self) -> List[float]:
        defaults: List[float] = []
        for v in self.prog.vars:
            if v.name in self.var_slots:
                raise self.err(f"state variable '{v.name}' is declared twice", v)
            if v.name in self.prog.params:
                raise self.err(
                    f"'{v.name}' is both a param and a state variable", v,
                    "rename one of them")
            rec = self.prog.records.get(v.type_name)
            count = v.array_len if v.array_len > 0 else 1
            stride = len(rec.fields) if rec is not None else 1
            n = count * stride
            base = self.alloc(n, v)
            if rec is not None:
                if v.default is not None:
                    raise self.err(
                        f"record '{v.name}' cannot have a default", v,
                        "records start zeroed; assign fields in 'init'")
                self.var_records[v.name] = (base, count, stride, rec.name)
            self.var_slots[v.name] = (base, n)
            self.scopes[0][v.name] = base
            self.decls.append((v.name, base, n, 1 if v.type_name in ("int", "bool") else 0))

            block = [0.0] * n
            if v.default is not None:
                if _literal(v.default) is not None:
                    if n > 1:
                        raise self.err(f"array '{v.name}' needs a [...] default", v)
                    block[0] = _literal(v.default)
                elif isinstance(v.default, ArrayLit):
                    if len(v.default.items) != n:
                        raise self.err(
                            f"'{v.name}' declared with {n} elements but its default "
                            f"has {len(v.default.items)}", v)
                    for i, item in enumerate(v.default.items):
                        if _literal(item) is None:
                            raise self.err("array defaults must be number literals", v)
                        block[i] = _literal(item)
                else:
                    raise self.err(f"default for '{v.name}' must be a constant", v)
            defaults.extend(block)
        return defaults

    def compile_fn(self, fn: FuncDecl) -> None:
        if fn.name in BUILTIN_ARITY:
            raise self.err(f"'{fn.name}' is a built-in and cannot be redefined", fn)

        frame_base = self.nslots
        entry = self.here

        self.push_scope()
        for p in fn.params:
            self.declare(p, fn)
        self._current_fn = fn.name
        for stmt in fn.body:
            self.stmt(stmt)
        self._current_fn = None
        self.emit(PUSH_CONST, self.const(0.0))
        self.emit(RET)
        self.pop_scope()

        self.func_info[fn.name] = {
            "entry": entry,
            "frame_base": frame_base,
            "nparams": len(fn.params),
        }

    def check_recursion(self) -> None:
        state: Dict[str, int] = {}

        def visit(name: str, chain: List[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                loop = " -> ".join(chain + [name])
                raise HunchError(
                    f"recursive call detected ({loop}); Hunch does not support recursion",
                    0, 0, self.prog.path)
            state[name] = 1
            for nxt in self.call_graph.get(name, ()):
                visit(nxt, chain + [name])
            state[name] = 2

        for name in list(self.call_graph):
            visit(name, [])

    def check_reward_agents(self) -> None:
        # A reward may come before its agent's hunch in the file, so this can
        # only be judged once everything is compiled. Without it a misspelt
        # name quietly becomes an agent that never decides, and the payoff
        # goes nowhere.
        deciders = [a for a in self.agents if a in self.agent_obs]
        for node in self.named_rewards:
            if node.agent in self.agent_obs:
                continue
            close = nearest(node.agent, deciders)
            if close:
                hint = f"did you mean '{close}'?"
            elif deciders:
                hint = "the hunch agents in this world are " + ", ".join(deciders)
            else:
                hint = (f"this world has no hunch; add one, e.g. "
                        f"let a = hunch {node.agent}(obs, 2);")
            raise self.err(
                f"reward names '{node.agent}', but no hunch in this world has that name",
                node.agent_at or node, hint)

    def resolve_bare_rewards(self) -> None:
        if not self.bare_rewards:
            return
        if len(self.agents) != 1:
            raise HunchError(
                "reward(x) is ambiguous in a world with "
                f"{len(self.agents)} hunch agents; write reward(agent_name, x)",
                0, 0, self.prog.path)
        for ip in self.bare_rewards:
            self.code[ip].a = 0

    # -- statements -------------------------------------------------------
    def stmt(self, node: Node) -> None:
        if isinstance(node, Let):
            if node.array_len > 0:
                base = self.declare_array(node.name, node.array_len, node)
                if node.value is not None:
                    if not isinstance(node.value, ArrayLit):
                        raise self.err(
                            f" '{node.name}' is an array; give it a [...] value", node)
                    if len(node.value.items) != node.array_len:
                        raise self.err(
                            f"'{node.name}' has {node.array_len} elements but the "
                            f"literal has {len(node.value.items)}", node)
                    for i, item in enumerate(node.value.items):
                        self.expr(item)
                        self.emit(STORE_VAR, base + i)
                else:
                    # A local array starts from zero each time its `let` runs,
                    # like any other local, so nothing in it outlives the step.
                    self.emit(ZERO, base, node.array_len)
                return
            if node.value is None:
                raise self.err(f"let '{node.name}' needs a value", node)
            if isinstance(node.value, ArrayLit):
                raise self.err("array literals need a declared size, e.g. let x: float[3]", node)
            # The value first: `let c = c + 1` must not read the c it declares,
            # which would be whatever the last step left in that slot.
            self.expr(node.value)
            slot = self.declare(node.name, node)
            self.emit(STORE_VAR, slot)
            return

        if isinstance(node, Assert):
            self.n_asserts += 1
            self.expr(node.cond)
            self.emit(ASSERT, node.line, node.col)
            return

        if isinstance(node, Assign):
            if isinstance(node.target, Field):
                target = node.target
                base, count, stride, recname = self.record_of(target, target.base)
                off = self.field_offset(recname, target.field, target)
                self.expr(node.value)
                if target.index is None:
                    if count != 1:
                        raise self.err(
                            f"'{target.base}' is an array of records", target,
                            f"index it first, e.g. {target.base}[0].{target.field}")
                    self.emit(STORE_VAR, base + off)
                else:
                    self.expr(target.index)
                    if stride > 1:
                        self.emit(PUSH_CONST, self.const(float(stride)))
                        self.emit(MUL)
                    self.emit(STORE_IDX, base + off, (count - 1) * stride + 1,
                              self.pos(target))
                return
            if isinstance(node.target, Index):
                base, n = self.arr_slots(node.target.base, node)
                self.expr(node.value)
                self.expr(node.target.index)
                self.emit(STORE_IDX, base, n, self.pos(node.target))
                return
            name = node.target.name if isinstance(node.target, Name) else node.target
            if isinstance(node.value, ArrayLit):
                base, n = self.arr_slots(name, node)
                if len(node.value.items) != n:
                    raise self.err(
                        f"'{name}' has {n} elements but assignment has "
                        f"{len(node.value.items)}", node)
                for i, item in enumerate(node.value.items):
                    self.expr(item)
                    self.emit(STORE_VAR, base + i)
                return
            slot = self.lookup(name)
            if slot is None:
                raise self.err(f"assignment to undeclared variable '{name}'", node)
            self.expr(node.value)
            self.emit(STORE_VAR, slot)
            return

        if isinstance(node, If):
            self.expr(node.cond)
            j_false = self.emit(JUMP_FALSE, 0)
            self.push_scope()
            for s in node.then_body:
                self.stmt(s)
            self.pop_scope()
            if node.else_body:
                j_end = self.emit(JUMP, 0)
                self.patch(j_false, self.here)
                self.push_scope()
                for s in node.else_body:
                    self.stmt(s)
                self.pop_scope()
                self.patch(j_end, self.here)
            else:
                self.patch(j_false, self.here)
            return

        if isinstance(node, While):
            loop_start = self.here
            self.expr(node.cond)
            j_false = self.emit(JUMP_FALSE, 0)
            self.push_scope()
            for s in node.body:
                self.stmt(s)
            self.pop_scope()
            self.emit(JUMP, loop_start)
            self.patch(j_false, self.here)
            return

        if isinstance(node, For):
            self.push_scope()
            i_slot = self.declare(node.var, node)
            end_slot = self.alloc(1, node)
            self.expr(node.start)
            self.emit(STORE_VAR, i_slot)
            self.expr(node.end)
            self.emit(STORE_VAR, end_slot)

            loop_start = self.here
            self.emit(PUSH_VAR, i_slot)
            self.emit(PUSH_VAR, end_slot)
            self.emit(LT)
            j_false = self.emit(JUMP_FALSE, 0)
            for s in node.body:
                self.stmt(s)
            self.emit(PUSH_VAR, i_slot)
            self.emit(PUSH_CONST, self.const(1.0))
            self.emit(ADD)
            self.emit(STORE_VAR, i_slot)
            self.emit(JUMP, loop_start)
            self.patch(j_false, self.here)
            self.pop_scope()
            return

        if isinstance(node, Return):
            if self._current_fn is None:
                raise self.err("'return' is only allowed inside a function", node)
            if node.value is None:
                self.emit(PUSH_CONST, self.const(0.0))
            else:
                self.expr(node.value)
            self.emit(RET)
            return

        if isinstance(node, ExprStmt):
            self.expr(node.expr)
            self.emit(POP)
            return

        if isinstance(node, Reward):
            if node.agent is None:
                self.expr(node.value)
                ip = self.emit(REWARD, -1)
                self.bare_rewards.append(ip)
                return
            agent = self.agent_id(node.agent, node)
            self.named_rewards.append(node)
            self.expr(node.value)
            self.emit(REWARD, agent)
            return

        if isinstance(node, DoneStmt):
            self.emit(DONE)
            return

        raise self.err(f"unsupported statement {type(node).__name__}", node)

    def record_of(self, node: Node, name: str) -> Tuple[int, int, int, str]:
        if name not in self.var_records:
            where = f"state variable '{name}'"
            raise self.err(f"{where} is not a record", node,
                           "declare it as 'name: RecordName' or 'name: RecordName[N]'")
        return self.var_records[name]

    def field_offset(self, recname: str, field: str, node: Node) -> int:
        rec = self.prog.records[recname]
        for i, (fname, _) in enumerate(rec.fields):
            if fname == field:
                return i
        names = ", ".join(f for f, _ in rec.fields)
        raise self.err(f"record '{recname}' has no field '{field}'", node,
                       f"fields are {names}")

    def array_here(self, name: str) -> bool:
        """An array that is in scope. var_slots also remembers arrays whose
        block has ended; those must not be read from another block."""
        got = self.var_slots.get(name)
        return got is not None and got[1] > 1 and self.lookup(name) == got[0]

    def arr_slots(self, name: str, node: Node) -> Tuple[int, int]:
        if name not in self.var_slots:
            raise self.err(f"'{name}' is not an array", node)
        if self.var_slots[name][1] > 1 and not self.array_here(name):
            raise self.err(
                f"array '{name}' is not in scope here", node,
                "a `let` array lasts until the end of its block; declare it in "
                "'state' to keep it between steps")
        base, n = self.var_slots[name]
        if n <= 1:
            raise self.err(f"'{name}' is not an array", node)
        return base, n

    def agent_id(self, name: str, node: Node) -> int:
        if name in self.agent_idx:
            return self.agent_idx[name]
        if len(self.agents) >= MAX_AGENTS:
            raise self.err(f"too many hunch agents (max {MAX_AGENTS})", node)
        idx = len(self.agents)
        self.agents.append(name)
        self.agent_idx[name] = idx
        return idx

    # -- expressions ------------------------------------------------------
    def expr(self, node: Node) -> None:
        if isinstance(node, Num):
            self.emit(PUSH_CONST, self.const(node.value))
            return

        if isinstance(node, Name):
            if node.name == "true":
                self.emit(PUSH_CONST, self.const(1.0))
                return
            if node.name == "false":
                self.emit(PUSH_CONST, self.const(0.0))
                return
            if node.name in self.prog.params:
                self.emit(PUSH_CONST, self.const(self.prog.params[node.name]))
                return
            if node.name in self.prog.action_names:
                self.emit(PUSH_CONST, self.const(float(self.prog.action_names[node.name][0])))
                return
            if node.name in self.var_records:
                recname = self.var_records[node.name][3]
                first = self.prog.records[recname].fields[0][0]
                raise self.err(
                    f"record '{node.name}' used as a value", node,
                    f"read a field, e.g. {node.name}.{first}")
            if self.array_here(node.name):
                raise self.err(
                    f"array '{node.name}' used as a value", node,
                    f"index it, e.g. {node.name}[0]")
            slot = self.lookup(node.name)
            if slot is None:
                known = [n for scope in self.scopes for n in scope]
                close = nearest(node.name, known)
                hint = (f"did you mean '{close}'?" if close
                        else "declare it in 'state' or with 'let' before use")
                raise self.err(f"unknown variable '{node.name}'", node, hint)
            self.emit(PUSH_VAR, slot)
            return

        if isinstance(node, Field):
            base, count, stride, recname = self.record_of(node, node.base)
            off = self.field_offset(recname, node.field, node)
            if node.index is None:
                if count != 1:
                    raise self.err(
                        f"'{node.base}' is an array of records", node,
                        f"index it first, e.g. {node.base}[0].{node.field}")
                self.emit(PUSH_VAR, base + off)
            else:
                self.expr(node.index)
                if stride > 1:
                    self.emit(PUSH_CONST, self.const(float(stride)))
                    self.emit(MUL)
                self.emit(PUSH_IDX, base + off, (count - 1) * stride + 1,
                          self.pos(node))
            return

        if isinstance(node, Index):
            base, n = self.arr_slots(node.base, node)
            self.expr(node.index)
            self.emit(PUSH_IDX, base, n, self.pos(node))
            return

        if isinstance(node, BinOp):
            self.expr(node.left)
            self.expr(node.right)
            if node.op in ("/", "%"):
                # Where the operator is, so `hunch test` can report a zero divisor.
                self.emit(BIN_OPS[node.op], node.line, node.col)
            else:
                self.emit(BIN_OPS[node.op])
            return

        if isinstance(node, UnOp):
            self.expr(node.operand)
            self.emit(NEG if node.op == "-" else NOT)
            return

        if isinstance(node, ArrayLit):
            raise self.err("array literals are only valid as a hunch() observation", node)

        if isinstance(node, Call):
            self.call(node)
            return

        if isinstance(node, Hunch):
            self.hunch(node)
            return

        raise self.err(f"unsupported expression {type(node).__name__}", node)

    def call(self, node: Call) -> None:
        name = node.name
        if name in BUILTIN_ARITY:
            want = BUILTIN_ARITY[name]
            if len(node.args) != want:
                raise self.err(
                    f"'{name}' takes {want} argument(s), got {len(node.args)}", node,
                    f"the signature is {name}(" + ", ".join(["x"] * want) + ")")
            for arg in node.args:
                self.expr(arg)
            if name == "rand":
                # Where it is, so a walk over every outcome can say which draw
                # it cannot list.
                self.emit(RAND, node.line, node.col)
            elif name == "randint":
                self.emit(RANDINT, node.line, node.col)
            else:
                self.emit(BUILTIN_OP[name])
            return

        if name not in self.func_index:
            close = nearest(name, list(BUILTIN_ARITY) + list(self.func_index))
            hint = (f"did you mean '{close}'?" if close else
                    "define it with 'fn name(a, b) { ... }' or check the builtin list")
            raise self.err(f"unknown function '{name}'", node, hint)

        decl = next(fn for fn in self.prog.funcs if fn.name == name)
        if len(node.args) != len(decl.params):
            raise self.err(
                f"'{name}' takes {len(decl.params)} argument(s), got {len(node.args)}", node)

        if self._current_fn is not None:
            self.call_graph[self._current_fn].add(name)

        # Arguments go on the value stack, not the shared scratch buffer.
        # Scratch is also where a hunch stages its observation, so staging
        # arguments there let a helper call inside an observation overwrite the
        # values already staged before it.
        for arg in node.args:
            self.expr(arg)
        self.emit(CALL, self.func_index[name])

    def hunch(self, node: Hunch) -> None:
        if node.n_actions < 2 or node.n_actions > MAX_ACTIONS:
            raise self.err(f"hunch needs 2..{MAX_ACTIONS} actions", node,
                           "pass the size of the action space, e.g. hunch p(obs, 3)")

        agent = self.agent_id(node.agent, node)

        if isinstance(node.obs, list):
            if not node.obs:
                raise self.err("hunch needs at least one observation value", node)
            if len(node.obs) > MAX_OBS:
                raise self.err(f"observation has {len(node.obs)} dims (max {MAX_OBS})", node)
            for item in node.obs:
                self.expr(item)
            obs_len = len(node.obs)
            labels = list(node.obs_labels)
        elif isinstance(node.obs, str):
            if self.array_here(node.obs):
                base, n = self.var_slots[node.obs]
                if n > MAX_OBS:
                    raise self.err(f"observation array has {n} dims (max {MAX_OBS})", node)
                # Read the array now, before the mask is worked out: a mask
                # may call a helper that writes to it.
                for i in range(n):
                    self.emit(PUSH_VAR, base + i)
                obs_len = n
                labels = [f"{node.obs}[{i}]" for i in range(n)]
            else:
                self.expr(Name(line=node.line, name=node.obs))
                obs_len = 1
                labels = [node.obs]
        elif node.obs_expr is not None:
            self.expr(node.obs_expr)
            obs_len = 1
            labels = list(node.obs_labels)
        else:
            raise self.err("hunch needs an observation", node)

        prev = self.agent_obs.get(node.agent)
        if prev is not None and prev != obs_len:
            raise self.err(
                f"agent '{node.agent}' used observation lengths {prev} and {obs_len}; "
                "keep them consistent", node)
        self.agent_obs[node.agent] = obs_len

        prev_act = self.agent_acts.get(node.agent)
        if prev_act is not None and prev_act != node.n_actions:
            raise self.err(
                f"agent '{node.agent}' used {prev_act} and {node.n_actions} actions; "
                "keep them consistent", node)
        self.agent_acts[node.agent] = node.n_actions

        # The first hunch of an agent names its observation; a later one that
        # passes something else in a slot makes that slot's label "a | b".
        seen = self.agent_obs_labels.setdefault(node.agent, labels)
        for i, label in enumerate(labels):
            if label and label not in seen[i].split(" | "):
                seen[i] = f"{seen[i]} | {label}" if seen[i] else label
        if node.action_names is not None:
            prev_names = self.agent_action_names.get(node.agent)
            if prev_names is not None and prev_names != node.action_names:
                raise self.err(
                    f"agent '{node.agent}' was given two different action lists", node,
                    "give the same names in the same order, or just the number")
            self.agent_action_names[node.agent] = node.action_names

        if node.legal is not None:
            # The VM holds the mask in one pending slot for the HUNCH that
            # follows; only the moves into the observation buffer come between.
            self.legal_mask(node)
            self.emit(LEGAL, node.n_actions, node.line, node.col)
        for i in reversed(range(obs_len)):
            self.emit(SCRATCH, i)
        self.emit(HUNCH, agent, obs_len, node.n_actions)

    def legal_mask(self, node: Hunch) -> None:
        """Push one value per action, action 0 first; nonzero means legal."""
        n = node.n_actions
        mask = node.legal
        if isinstance(mask, ArrayLit):
            if len(mask.items) != n:
                raise self.err(
                    f"legal-move mask has {len(mask.items)} value(s) but hunch "
                    f"'{node.agent}' has {n} actions", mask,
                    "give exactly one value per action; nonzero means legal")
            for item in mask.items:
                self.expr(item)
            return

        name = mask.name
        slot = self.lookup(name)
        if slot is None and name not in self.prog.params:
            arrays = [k for scope in self.scopes for k in scope
                      if self.var_slots.get(k, (0, 0))[1] == n]
            close = nearest(name, arrays)
            hint = (f"did you mean '{close}'?" if close else
                    f"declare it first, e.g. let {name}: int[{n}]; or pass a [...] list")
            raise self.err(f"unknown legal-move mask '{name}'", mask, hint)
        if name in self.var_records:
            raise self.err(
                f"record '{name}' cannot be a legal-move mask", mask,
                f"use an array of {n} numbers, one per action")
        # var_slots also remembers arrays whose scope has ended, so trust it
        # only when it describes the variable that is in scope here.
        base, length = self.var_slots.get(name, (-1, 0))
        if slot is None or base != slot or length <= 1:
            raise self.err(
                f"'{name}' is not an array, so it cannot be a legal-move mask", mask,
                f"a mask has one value per action, e.g. let ok: int[{n}]; "
                "or pass a [...] list")
        if length != n:
            raise self.err(
                f"legal-move mask '{name}' has {length} elements but hunch "
                f"'{node.agent}' has {n} actions", mask,
                "the mask needs exactly one element per action")
        for i in range(n):
            self.emit(PUSH_VAR, base + i)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

MAGIC = b"HUNCHBC\x01"
VERSION = 1


def _wstr(buf: bytearray, s: str) -> None:
    data = s.encode("utf-8")
    buf += struct.pack("<I", len(data))
    buf += data


def serialize(unit: dict) -> bytes:
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<I", VERSION)

    agents: List[str] = unit["agents"]
    decls = unit["decls"]
    funcs = unit["funcs"]
    code: List[Insn] = unit["code"]
    consts: List[float] = unit["consts"]

    buf += struct.pack("<IIIIIII", unit["nslots"], len(consts), len(code),
                       len(agents), len(funcs), unit["horizon"], len(decls))
    buf += struct.pack("<II", unit["init_entry"], unit["step_entry"])

    _wstr(buf, unit["world"])
    _wstr(buf, unit.get("path", ""))

    for name in agents:
        _wstr(buf, name)
        buf += struct.pack("<II", unit["agent_obs"].get(name, 1),
                           unit["agent_acts"].get(name, 2))

    for (name, base, length, is_int) in decls:
        _wstr(buf, name)
        buf += struct.pack("<III", base, length, is_int)

    for name in funcs:
        f = funcs[name]
        _wstr(buf, name)
        buf += struct.pack("<III", f["entry"], f["frame_base"], f["nparams"])

    if consts:
        buf += struct.pack(f"<{len(consts)}d", *consts)

    flat: List[int] = []
    for ins in code:
        flat.extend((ins.op, ins.a, ins.b, ins.c))
    if flat:
        buf += struct.pack(f"<{len(flat)}i", *flat)

    defaults: List[float] = unit["defaults"]
    if defaults:
        buf += struct.pack(f"<{len(defaults)}d", *defaults)

    return bytes(buf)


def compile_program(prog: Program, params: dict = None) -> Tuple[bytes, dict]:
    if params:
        unknown = sorted(set(params) - set(prog.params))
        if unknown:
            raise HunchError(
                f"unknown param '{unknown[0]}'", 0, 0, prog.path,
                "this world declares "
                + (", ".join(sorted(prog.params)) or "no params"))
        prog.params.update(params)
    unit = CodeGen(prog).compile()
    return serialize(unit), unit
