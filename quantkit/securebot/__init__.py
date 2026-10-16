"""Project 6: a paper-trading bot built with security and failure-safety first."""
from .audit import AuditLog, redact
from .bot import TradingBot
from .broker import Broker, Order, PaperBroker
from .config import MissingSecret, Secrets, load_secrets, scan_for_secrets
from .guard import GuardRejected, KillSwitch, RiskGuard

__all__ = ["AuditLog", "redact", "TradingBot", "Broker", "Order", "PaperBroker", "MissingSecret",
           "Secrets", "load_secrets", "scan_for_secrets", "GuardRejected", "KillSwitch", "RiskGuard"]
