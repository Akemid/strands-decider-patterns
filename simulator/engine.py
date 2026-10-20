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
from typing import Any

from simulator.fake_ctf_mcp import FIELDS, FLAG_BASES, GRID, WALLS, make_route

Cell = tuple[int, int]
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ENEMY_CHASE_RANGE = 3
THREAT_RANGE = 6
RISK_LOOKAHEAD = 12
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

    def _free(self, cell: Cell, blocked: set[Cell]) -> bool:
        in_grid = 0 <= cell[0] < GRID["width"] and 0 <= cell[1] < GRID["height"]
        return in_grid and cell not in self.walls and cell not in blocked

    def _path(self, start: Cell, goal: Cell, blocked: set[Cell]) -> list[Cell] | None:
        """Shortest path as the cells after `start` up to and including `goal` (BFS, fixed direction order)."""
        if start == goal:
            return []
        seen = {start: None}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            for dx, dy in DELTAS.values():
                nxt = (here[0] + dx, here[1] + dy)
                if nxt in seen or not (self._free(nxt, blocked) or nxt == goal):
                    continue
                seen[nxt] = here
                if nxt == goal:
                    cells = [nxt]
                    while seen[cells[-1]] != start:
                        cells.append(seen[cells[-1]])
                    return cells[::-1]
                queue.append(nxt)
        return None

    @staticmethod
    def _direction(start: Cell, nxt: Cell) -> str:
        return next(name for name, (dx, dy) in DELTAS.items() if (start[0] + dx, start[1] + dy) == nxt)

    # ---- actions ---------------------------------------------------------

    def apply(self, action: str, args: dict[str, Any]) -> None:
        """Apply an accepted action immediately; enemies react in `end_tick`."""
        if action == "wait":
            return
        if action == "move":
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
            path = self._path(cell, self.position, blocked - {self.position})
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

    def _agent(self, aid: str, team: str, cell: Cell, carrying: bool) -> dict[str, Any]:
        return {"id": aid, "team": team, "position": _xy(cell), "carrying": "flag" if carrying else None,
                "disconnected": False}

    def _route(self, rid: str, goal: str, path: list[Cell]) -> dict[str, Any]:
        steps = list(enumerate(path[:RISK_LOOKAHEAD], start=1))
        enemy_team = OTHER[self.team]
        risks = tuple(
            (step, eid) for step, cell in steps if self._in_field(cell, enemy_team)
            for eid, ecell in self.enemies.items() if _dist(cell, ecell) <= 1
        )
        near = min(((_dist(c, self.position), eid) for eid, c in self.enemies.items()), default=None)
        threat = (near[1], near[0], near[0]) if near and near[0] <= THREAT_RANGE else None
        first = self._direction(self.position, path[0]) if path else None
        blocked = bool(path) and path[0] in self._occupied()
        route = make_route(rid, goal, first, max(0, len(path) - (1 if goal == "tag" else 0)), risks=risks,
                           blocked=blocked, threat=threat, path=[_xy(c) for c in path])
        if goal == "tag" and len(path) <= 1:
            route["first_step"] = None
        return route

    def routes(self) -> list[dict[str, Any]]:
        mine, theirs = self.bases[self.team], self.bases[OTHER[self.team]]
        out: list[dict[str, Any]] = []
        occupied = self._occupied()
        if self.carrying:
            path = self._path(self.position, mine, occupied)
            if path is not None:
                out.append(self._route("own_base_direct", "own_base", path))
        else:
            direct = self._path(self.position, theirs, occupied)
            if direct is not None:
                out.append(self._route("enemy_flag_direct", "enemy_flag", direct))
                avoid = occupied | {(c[0] + dx, c[1] + dy) for c in self.enemies.values()
                                    for dx, dy in (*DELTAS.values(), (0, 0))}
                detour = self._path(self.position, theirs, avoid - {theirs, self.position})
                if detour is not None and detour != direct:
                    out.append(self._route("enemy_flag_detour", "enemy_flag", detour))
        intruders = {e: c for e, c in self.enemies.items() if self._in_field(c, self.team)}
        for eid, cell in intruders.items():
            path = self._path(self.position, cell, occupied - {cell})
            if path is not None and self._in_field(self.position, self.team):
                out.append(self._route(f"tag:{eid}", "tag", path))
        if intruders:
            path = self._path(self.position, mine, occupied)
            if path is not None:
                out.append(self._route("defend_flag", "defend_flag", path))
        return out

    def state(self, tick: int) -> dict[str, Any]:
        """The turn `state` for the controlled agent (v2 shape: no static map keys)."""
        enemy_team = OTHER[self.team]
        moves = {}
        for name, (dx, dy) in DELTAS.items():
            cell = (self.position[0] + dx, self.position[1] + dy)
            moves[name] = ("open" if self._free(cell, self._occupied())
                           else "agent" if cell in self._occupied() else "wall")
        enemy_flag = self.position if self.carrying else self.bases[enemy_team]
        agents = [self._agent(self.agent_id, self.team, self.position, self.carrying)]
        agents += [self._agent(a, self.team, c, False) for a, c in self.teammates.items()]
        agents += [self._agent(a, enemy_team, c, False) for a, c in self.enemies.items()]
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
            "teammates": [{"id": a, "position": _xy(c)} for a, c in self.teammates.items()],
            "moves": moves,
            "routes": self.routes(),
        }
