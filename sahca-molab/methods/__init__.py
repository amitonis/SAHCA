"""Credit assignment method registry."""

from typing import Any

from methods.base import CreditAssignmentMethod
from methods.grpo import GRPO
from methods.gigpo import GiGPO
from methods.gagpo import GAGPO
from methods.sahca import SAHCA


_METHODS = {
    "grpo": GRPO,
    "gigpo": GiGPO,
    "gagpo": GAGPO,
    "sahca": SAHCA,
}


def get_method(name: str, **kwargs: Any) -> CreditAssignmentMethod:
    """Instantiate a credit assignment method by name.

    Args:
        name: Method name ('grpo', 'gigpo', 'gagpo', 'sahca').
        **kwargs: Passed to the method constructor (e.g., alpha, beta, gamma
                  for SAHCA, ignored for methods that don't accept them).

    Returns:
        Instance of the requested CreditAssignmentMethod.

    Raises:
        ValueError: If name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in _METHODS:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(_METHODS.keys())}"
        )

    cls = _METHODS[name_lower]

    # Only pass kwargs if the constructor accepts them (SAHCA does, others don't)
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
