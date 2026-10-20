"""A deliberately tiny Capture the Flag game, just enough for an agent to play a match to `match_over`.

It is a simulator, not the real game: one controlled agent, two slow chasing enemies, static teammates, a seeded
random walk, and simplified route risk. Rules:

- the agent moves one cell per action (walls and other agents block);
- `grab_flag` on the enemy base picks the flag up; carrying it onto your own base scores and ends the match;
- an enemy standing next to you while you are inside ITS field tags you: you respawn at your base, dropping the flag;
- `tag` sends an adjacent enemy back to its base, but only while you stand in your own field;
- after `max_ticks` the match ends in a tie.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from typing import Any

from simulator.fake_ctf_mcp import FIELDS, FLAG_BASES, GRID, WALLS, make_route

Cell = tuple[int, int]
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ENEMY_CHASE_RANGE = 3
MAX_PATH_CELLS = 8
MAX_ROUTES = 7
DEFEND_RADIUS = 4
OTHER = {"red": "blue", "blue": "red"}


def _cell(c: dict) -> Cell:
    return c["x"], c["y"]


def _xy(c: Cell) -> dict[str, int]:
    return {"x": c[0], "y": c[1]}


def _dist(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SimMatch:
    def __init__(self, *, max_ticks: int = 60, seed: int = 0) -> None:
        self.max_ticks = max_ticks
        self.rng = random.Random(seed)
        self.walls: set[Cell] = {_cell(w) for w in WALLS}
        self.bases = {t: _cell(c) for t, c in FLAG_BASES.items()}
        self.tick = 0
        self.over = False
        self.winner: str | None = None
        self.score = {"red": 0, "blue": 0}
        self.team = "red"
        self.agent_id = "red-1"
        self.position: Cell = self.bases["red"]
        self.carrying = False
        self.tagged_count = 0
        self.teammates: dict[str, Cell] = {}
        self.enemies: dict[str, Cell] = {}

    # ---- setup -----------------------------------------------------------

    def join(self, team: str, index: int) -> str:
        """Seat the controlled agent and lay out the other agents. Returns the agent id."""
        self.team, self.agent_id = team, f"{team}-{index + 1}"
        mine, theirs = self.bases[team], self.bases[OTHER[team]]
        self.position = mine
        self.teammates = {
            f"{team}-{i + 1}": (mine[0], mine[1] + offset)
            for i, offset in zip([j for j in range(3) if j != index], (-3, 3), strict=True)
        }
        self.enemies = {
            f"{OTHER[team]}-1": (theirs[0] - 1, theirs[1] - 4),
            f"{OTHER[team]}-2": (theirs[0] - 1, theirs[1] + 4),
        }
        return self.agent_id

    # ---- helpers ---------------------------------------------------------

    def _field(self, team: str) -> tuple[int, int]:
        f = FIELDS[team]
        return f["min_x"], f["max_x"]

    def _in_field(self, cell: Cell, team: str) -> bool:
        lo, hi = self._field(team)
        return lo <= cell[0] <= hi

    def _occupied(self) -> set[Cell]:
        return set(self.enemies.values()) | set(self.teammates.values())

    def _walkable(self, cell: Cell) -> bool:
        """Inside the grid and not a wall. Agents never count: they do not block pathfinding."""
        return 0 <= cell[0] < GRID["width"] and 0 <= cell[1] < GRID["height"] and cell not in self.walls

    def _free(self, cell: Cell, blocked: set[Cell]) -> bool:
        return self._walkable(cell) and cell not in blocked

    def _path(
        self,
        start: Cell,
        is_goal: Callable[[Cell], bool],
        passable: Callable[[Cell], bool],
        forbid_first: frozenset[Cell] = frozenset(),
    ) -> list[Cell] | None:
        """Shortest path as the cells after `start` up to the first goal cell (BFS, fixed direction order).

        `[]` when `start` is already a goal. `forbid_first` lists cells that may not be the first step.
        """
        if is_goal(start):
            return []
        parent: dict[Cell, Cell | None] = {start: None}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            for dx, dy in DELTAS.values():
                nxt = (here[0] + dx, here[1] + dy)
                if nxt in parent or not passable(nxt) or (here == start and nxt in forbid_first):
                    continue
                parent[nxt] = here
                if is_goal(nxt):
                    cells = [nxt]
                    while parent[cells[-1]] != start:
                        cells.append(parent[cells[-1]])  # type: ignore[arg-type]
                    return cells[::-1]
                queue.append(nxt)
        return None

    def _goal_path(self, start: Cell, goal: Cell, **kw: Any) -> list[Cell] | None:
        return self._path(start, lambda c: c == goal, self._walkable, **kw)

    @staticmethod
    def _direction(start: Cell, nxt: Cell) -> str:
        return next(name for name, (dx, dy) in DELTAS.items() if (start[0] + dx, start[1] + dy) == nxt)

    def apply(self, action: str, args: dict[str, Any]) -> None:
        """Apply an accepted action immediately; enemies react in `end_tick`."""
        if action == "wait":
            return
        if action == "move":
            if args.get("direction") not in DELTAS:
                raise ValueError(f"invalid direction: {args.get('direction')!r}")
            dx, dy = DELTAS[args["direction"]]
            target = (self.position[0] + dx, self.position[1] + dy)
            if self._free(target, self._occupied()):
                self.position = target
        elif action == "grab_flag":
            if self.position == self.bases[OTHER[self.team]]:
                self.carrying = True
        elif action == "tag":
            enemy = self.enemies.get(args["enemy_id"])
            if enemy and _dist(enemy, self.position) == 1 and self._in_field(self.position, self.team):
                self.enemies[args["enemy_id"]] = self.bases[OTHER[self.team]]
        else:
            raise KeyError(action)

    def end_tick(self) -> None:
        """Resolve the tick: capture, enemy movement, tags, and the tick limit."""
        self.tick += 1
        if self.carrying and self.position == self.bases[self.team]:
            self.score[self.team] += 1
            self.over, self.winner = True, self.team
            return
        if self.tick % 2 == 0:  # enemies are slower than the agent: one step every other tick
            for eid, cell in list(self.enemies.items()):
                self.enemies[eid] = self._enemy_step(eid, cell)
        enemy_team = OTHER[self.team]
        near = any(_dist(c, self.position) == 1 for c in self.enemies.values())
        if near and self._in_field(self.position, enemy_team):
            self.position, self.carrying = self.bases[self.team], False
            self.tagged_count += 1
        if self.tick >= self.max_ticks:
            self.over, self.winner = True, "tie"

    def _enemy_step(self, eid: str, cell: Cell) -> Cell:
        blocked = (self._occupied() - {cell}) | {self.position}
        if _dist(cell, self.position) <= ENEMY_CHASE_RANGE:
            walk = blocked - {self.position}
            path = self._path(cell, lambda c: c == self.position, lambda c: self._free(c, walk))
            if path and len(path) > 1:
                return path[0]
            return cell
        options = [
            (cell[0] + dx, cell[1] + dy)
            for dx, dy in DELTAS.values()
            if self._free((cell[0] + dx, cell[1] + dy), blocked)
            and self._in_field((cell[0] + dx, cell[1] + dy), OTHER[self.team])
        ]
        return self.rng.choice(options) if options else cell

    # ---- observation -----------------------------------------------------

    def _agent(self, aid: str, team: str, cell: Cell, carrying: str | None) -> dict[str, Any]:
        return {"id": aid, "team": team, "position": _xy(cell), "carrying": carrying, "disconnected": False}

    def _threat_field(self) -> dict[Cell, tuple[int, str]]:
        """Grid distance from every reachable cell to the nearest enemy standing in its own field.

        Multi-source BFS in enemy-id order, so equidistant enemies resolve to the lowest id.
        """
        field: dict[Cell, tuple[int, str]] = {}
        queue: deque[Cell] = deque()
        for eid, cell in sorted(self._threat_enemies().items()):
            if cell not in field:
                field[cell] = (0, eid)
                queue.append(cell)
        while queue:
            here = queue.popleft()
            dist, eid = field[here]
            for dx, dy in DELTAS.values():
                nxt = (here[0] + dx, here[1] + dy)
                if self._walkable(nxt) and nxt not in field:
                    field[nxt] = (dist + 1, eid)
                    queue.append(nxt)
        return field

    def _threat_enemies(self) -> dict[str, Cell]:
        return {e: c for e, c in self.enemies.items() if self._in_field(c, OTHER[self.team])}

    def _risk_cells(self) -> dict[Cell, list[str]]:
        """Walkable cells next to an enemy standing in its own field, with the enemies that threaten them."""
        risks: dict[Cell, list[str]] = {}
        for eid, cell in sorted(self._threat_enemies().items()):
            for dx, dy in DELTAS.values():
                near = (cell[0] + dx, cell[1] + dy)
                if self._walkable(near):
                    risks.setdefault(near, []).append(eid)
        return risks

    def _route(self, rid: str, goal: str, path: list[Cell], risks: dict[Cell, list[str]],
               field: dict[Cell, tuple[int, str]]) -> dict[str, Any]:
        route_risks = tuple((step, eid, cell) for step, cell in enumerate(path, start=1)
                            for eid in risks.get(cell, []))
        threat: tuple[str, int, int] | None = None
        for step, cell in enumerate([self.position, *path]):
            if cell in field and (threat is None or field[cell][0] < threat[2]):
                threat = (field[cell][1], step, field[cell][0])
        first = path[0] if path else None
        occupied = first is not None and first in self._occupied()
        return make_route(rid, goal, self._direction(self.position, first) if first else None, len(path),
                          risks=route_risks, blocked_reason="occupied" if occupied else None, threat=threat,
                          path=[_xy(c) for c in path[:MAX_PATH_CELLS]])

    def _flag_family(self, prefix: str, goal: str, dest: Cell, risks: dict[Cell, list[str]],
                     field: dict[Cell, tuple[int, str]]) -> list[dict[str, Any]]:
        """`_direct`, `_safe` (no risk cell on the way) and `_detour` (a different first step) routes to `dest`."""
        direct = self._goal_path(self.position, dest)
        if direct is None:
            return []
        safe = self._path(self.position, lambda c: c == dest, lambda c: self._walkable(c) and c not in risks)
        firsts = lambda *paths: frozenset(p[0] for p in paths if p)  # noqa: E731
        detour = None
        if direct:
            detour = (self._goal_path(self.position, dest, forbid_first=firsts(direct, safe))
                      or self._goal_path(self.position, dest, forbid_first=firsts(direct)))
        family, seen = [], set()
        for suffix, path in (("direct", direct), ("safe", safe), ("detour", detour)):
            if path is not None and tuple(path) not in seen:
                seen.add(tuple(path))
                family.append(self._route(f"{prefix}_{suffix}", goal, path, risks, field))
        return family

    def routes(self) -> list[dict[str, Any]]:
        mine, theirs = self.bases[self.team], self.bases[OTHER[self.team]]
        risks, field = self._risk_cells(), self._threat_field()
        enemy_flag = self.position if self.carrying else theirs
        out = self._flag_family("enemy_flag", "enemy_flag", enemy_flag, risks, field)
        if self.carrying:  # a teammate carrying would count too, but static teammates never do
            out += self._flag_family("own_base", "own_base", mine, risks, field)
        for eid, cell in sorted(self.enemies.items()):
            if not self._in_field(cell, self.team):
                continue
            near = {(cell[0] + dx, cell[1] + dy) for dx, dy in DELTAS.values()}
            path = self._path(self.position, lambda c, near=near: c in near and self._in_field(c, self.team),
                              lambda c, cell=cell: self._walkable(c) and c != cell)
            if path is not None:
                out.append(self._route(f"tag:{eid}", "tag", path, risks, field))
        if any(self._steps_between(mine, c) <= DEFEND_RADIUS for c in self.enemies.values()):
            path = self._goal_path(self.position, mine)
            if path is not None:
                out.append(self._route("defend_flag", "defend_flag", path, risks, field))
        return sorted(out, key=lambda r: r["id"])[:MAX_ROUTES]

    def _steps_between(self, a: Cell, b: Cell) -> int | float:
        path = self._goal_path(a, b)
        return float("inf") if path is None else len(path)

    def state(self, tick: int) -> dict[str, Any]:
        """The turn `state` for the controlled agent (v2 shape: no static map keys)."""
        enemy_team = OTHER[self.team]
        moves = {}
        for name, (dx, dy) in DELTAS.items():
            cell = (self.position[0] + dx, self.position[1] + dy)
            in_grid = 0 <= cell[0] < GRID["width"] and 0 <= cell[1] < GRID["height"]
            moves[name] = "out_of_bounds" if not in_grid else "wall" if cell in self.walls else "open"
        enemy_flag = self.position if self.carrying else self.bases[enemy_team]
        agents = [self._agent(self.agent_id, self.team, self.position, enemy_team if self.carrying else None)]
        agents += [self._agent(a, self.team, c, None) for a, c in self.teammates.items()]
        agents += [self._agent(a, enemy_team, c, None) for a, c in self.enemies.items()]
        return {
            "tick": tick,
            "self": self.agent_id,
            "self_position": _xy(self.position),
            "agents": agents,
            "flags": [
                {"team": self.team, "position": _xy(self.bases[self.team]), "carried_by": None},
                {"team": enemy_team, "position": _xy(enemy_flag), "carried_by": self.agent_id if self.carrying else None},
            ],
            "score": dict(self.score),
            "teammates": list(self.teammates),
            "moves": moves,
            "routes": self.routes(),
        }
