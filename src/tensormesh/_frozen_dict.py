"""Immutable dict subclass that is compatible with torch.compile."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class FrozenDict[K, V](dict[K, V]):
    """Immutable dict that `torch.compile` can trace through.

    Unlike `frozendict.frozendict`, this class does **not** override `__new__`, which
    lets `torch._dynamo` treat it as an ordinary `dict` subclass.
    """

    def __init__(self, mapping_or_iterable: Mapping[K, V] | None = None, /) -> None:
        if mapping_or_iterable is None:
            super().__init__()
        else:
            super().__init__(mapping_or_iterable)

    # -- block all mutation -----------------------------------------------------

    def _raise(self) -> NoReturn:
        msg = f"'{type(self).__name__}' object does not support mutation"
        raise TypeError(msg)

    def __setitem__(self, key: K, value: V) -> None:
        self._raise()

    def __delitem__(self, key: K) -> None:
        self._raise()

    def clear(self) -> None:
        self._raise()

    def pop(self, *_args: object) -> V:
        self._raise()

    def popitem(self) -> tuple[K, V]:
        self._raise()

    def setdefault(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        key: K,  # noqa: ARG002
        default: V,  # noqa: ARG002
    ) -> V:
        self._raise()

    def update(self, *_args: object, **_kwargs: V) -> None:
        self._raise()

    def __ior__(self, other: object) -> NoReturn:  # noqa: PYI034
        self._raise()

    # -- immutable copy helpers -------------------------------------------------

    def __or__(  # type: ignore[override]
        self, other: Mapping[K, V]
    ) -> FrozenDict[K, V]:
        return FrozenDict(dict.__or__(self, other))  # type: ignore[arg-type]

    def __ror__(  # type: ignore[override]
        self, other: Mapping[K, V]
    ) -> FrozenDict[K, V]:
        return FrozenDict(dict.__ror__(self, other))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"FrozenDict({dict.__repr__(self)})"

    def __iter__(self) -> Iterator[K]:
        return dict.__iter__(self)
