"""Pre-trade safety checks and a kill switch."""
from __future__ import annotations

import os
import time
from collections import deque


class GuardRejected(Exception):
    pass


class KillSwitch:
    """Trading halts if the flag file exists or `trip()` was called. Create the file to stop the bot by hand."""

    def __init__(self, flag_path: str | None = None):
        self.flag_path = flag_path
        self._tripped = False
        self.reason = ""

    def trip(self, reason: str) -> None:
        self._tripped, self.reason = True, reason

    @property
    def tripped(self) -> bool:
        return self._tripped or bool(self.flag_path and os.path.exists(self.flag_path))


class RiskGuard:
    def __init__(self, kill_switch: KillSwitch, max_order_qty: int = 10, max_position: int = 50,
                 max_orders_per_min: int = 20, daily_loss_limit: float = 0.02, clock=time.time):
        self.ks = kill_switch
        self.max_order_qty = max_order_qty
        self.max_position = max_position
        self.max_orders_per_min = max_orders_per_min
        self.daily_loss_limit = daily_loss_limit
        self._clock = clock
        self._recent: deque[float] = deque()

    def check(self, qty: int, side: str, position: int, daily_pnl_frac: float) -> None:
        if self.ks.tripped:
            raise GuardRejected(f"kill switch active {self.ks.reason}".strip())
        if daily_pnl_frac <= -self.daily_loss_limit:
            self.ks.trip("daily loss limit")
            raise GuardRejected("daily loss limit hit; kill switch tripped")
        if qty <= 0 or qty > self.max_order_qty:
            raise GuardRejected(f"order qty {qty} outside 1..{self.max_order_qty}")
        new_pos = position + qty if side.upper() == "BUY" else position - qty
        if abs(new_pos) > self.max_position:
            raise GuardRejected(f"resulting position {new_pos} exceeds limit {self.max_position}")
        now = self._clock()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if len(self._recent) >= self.max_orders_per_min:
            raise GuardRejected("order rate limit exceeded")
        self._recent.append(now)
