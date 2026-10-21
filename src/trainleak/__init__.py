"""trainleak: does your model leak its training data?"""

from .audit import ALL_ATTACKS, AuditReport, audit
from .metrics import AttackResult

__all__ = ["audit", "AuditReport", "AttackResult", "ALL_ATTACKS"]
__version__ = "0.1.0"
