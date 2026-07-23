from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator, TypeVar

T = TypeVar("T")


def run_sync(coro):
    """Run an async coroutine from a sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def async_iter_to_sync(async_iterator: AsyncIterator[T]) -> Iterator[T]:
    """Convert an async iterator into a sync iterator."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                yield loop.run_until_complete(async_iterator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()
        asyncio.set_event_loop(None)
