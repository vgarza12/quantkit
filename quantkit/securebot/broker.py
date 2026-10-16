"""Broker interface plus a paper broker that can simulate nasty network failures."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    client_order_id: str   # idempotency key: same id must never create two orders
    symbol: str
    side: str              # "BUY" / "SELL"
    qty: int
    price_hint: float


class Broker(ABC):
    @abstractmethod
    def submit(self, order: Order) -> dict: ...


class PaperBroker(Broker):
    """Simulated broker.

    fail_first_n:        raise ConnectionError BEFORE the order is received.
    drop_response_n:     receive the order, then raise ConnectionError (response lost).
                         This is the dangerous case: a naive retry would double-trade.
    """

    def __init__(self, slippage_bps: float = 1.0, seed: int = 0, fail_first_n: int = 0, drop_response_n: int = 0):
        self._rng = random.Random(seed)
        self.slippage_bps = slippage_bps
        self.fail_first_n = fail_first_n
        self.drop_response_n = drop_response_n
        self._calls = 0
        self.orders: dict[str, dict] = {}

    def submit(self, order: Order) -> dict:
        self._calls += 1
        if self._calls <= self.fail_first_n:
            raise ConnectionError("simulated: request never reached broker")
        if order.client_order_id in self.orders:            # idempotent: return the original
            return self.orders[order.client_order_id]
        sign = 1 if order.side.upper() == "BUY" else -1
        slip = self._rng.uniform(0, 2 * self.slippage_bps) / 1e4
        fill = {"client_order_id": order.client_order_id, "symbol": order.symbol, "side": order.side,
                "qty": order.qty, "fill_price": order.price_hint * (1 + sign * slip), "status": "FILLED"}
        self.orders[order.client_order_id] = fill
        if self._calls <= self.fail_first_n + self.drop_response_n:
            raise ConnectionError("simulated: order received but response lost")
        return fill
