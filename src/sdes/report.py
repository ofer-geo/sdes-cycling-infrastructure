"""Record how many rows each cleaning step removes, so every funnel is auditable."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FilterReport:
    name: str
    initial: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)

    def start(self, n: int) -> None:
        self.initial += n

    def add(self, step: str, removed: int) -> None:
        for i, (s, n) in enumerate(self.steps):
            if s == step:
                self.steps[i] = (s, n + removed)
                return
        self.steps.append((step, removed))

    @property
    def remaining(self) -> int:
        return self.initial - sum(n for _, n in self.steps)

    def to_frame(self) -> pd.DataFrame:
        rows, left = [("initial", 0, self.initial)], self.initial
        for step, removed in self.steps:
            left -= removed
            rows.append((step, removed, left))
        return pd.DataFrame(rows, columns=["step", "removed", "remaining"])

    def __str__(self) -> str:
        return f"{self.name}\n" + self.to_frame().to_string(index=False)
