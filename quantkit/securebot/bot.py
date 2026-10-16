"""Trading bot: guarded, idempotent, retry-safe, and fully audited."""
from __future__ import annotations

import time
import uuid

from .audit import AuditLog
from .broker import Broker, Order
from .guard import GuardRejected, RiskGuard


class TradingBot:
    def __init__(self, broker: Broker, guard: RiskGuard, audit: AuditLog, max_retries: int = 3,
                 backoff_s: float = 0.5, sleep=time.sleep):
        self.broker, self.guard, self.audit = broker, guard, audit
        self.max_retries, self.backoff_s, self._sleep = max_retries, backoff_s, sleep

    def place(self, symbol: str, side: str, qty: int, price_hint: float, position: int = 0,
              daily_pnl_frac: float = 0.0) -> dict | None:
        try:
            self.guard.check(qty, side, position, daily_pnl_frac)
        except GuardRejected as e:
            self.audit.append({"type": "rejected", "symbol": symbol, "side": side, "qty": qty, "reason": str(e)})
            return None

        # ONE id for all attempts: a retry after a lost response can never double-trade.
        order = Order(uuid.uuid4().hex, symbol, side, qty, price_hint)
        for attempt in range(1, self.max_retries + 1):
            try:
                fill = self.broker.submit(order)
                self.audit.append({"type": "filled", "attempt": attempt, **fill})
                return fill
            except ConnectionError as e:
                self.audit.append({"type": "retry", "attempt": attempt, "id": order.client_order_id, "error": str(e)})
                if attempt < self.max_retries:
                    self._sleep(self.backoff_s * 2 ** (attempt - 1))
        self.guard.ks.trip("broker unreachable after retries")
        self.audit.append({"type": "halt", "id": order.client_order_id, "reason": "broker unreachable"})
        return None
