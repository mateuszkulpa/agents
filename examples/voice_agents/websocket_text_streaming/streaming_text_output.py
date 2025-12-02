"""
StreamingTextOutput - A TextOutput implementation for real-time text streaming.

This module provides a TextOutput class that captures LLM text deltas in real-time,
making them available via an async iterator for streaming over WebSocket or similar protocols.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from livekit.agents.voice.io import TextOutput


@dataclass
class TextDeltaEvent:
    """Event emitted for each text chunk from the LLM."""

    delta: str
    """The new text chunk."""
    accumulated: str
    """All text accumulated so far."""
    type: Literal["text_delta"] = "text_delta"


@dataclass
class TextCompleteEvent:
    """Event emitted when the LLM generation is complete."""

    text: str
    """The complete generated text."""
    type: Literal["text_complete"] = "text_complete"


StreamEvent = TextDeltaEvent | TextCompleteEvent


class StreamingTextOutput(TextOutput):
    """
    A TextOutput implementation that streams text deltas in real-time.

    This class captures LLM output as it's generated and makes it available
    via an async iterator. Perfect for WebSocket streaming or similar use cases.

    Example:
        ```python
        streaming_output = StreamingTextOutput()
        sess.output.transcription = streaming_output

        # In a separate task, consume the stream
        async for event in streaming_output.stream():
            if event.type == "text_delta":
                await websocket.send_json({"delta": event.delta})
            elif event.type == "text_complete":
                await websocket.send_json({"complete": event.text})
        ```
    """

    def __init__(self, *, next_in_chain: TextOutput | None = None) -> None:
        super().__init__(label="streaming_text", next_in_chain=next_in_chain)
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._accumulated_text: str = ""
        self._closed: bool = False

    @property
    def accumulated_text(self) -> str:
        """Returns all text accumulated so far."""
        return self._accumulated_text

    async def capture_text(self, text: str) -> None:
        """
        Called by the agent framework for each text chunk.
        Emits a TextDeltaEvent to the stream.
        """
        self._accumulated_text += text

        event = TextDeltaEvent(delta=text, accumulated=self._accumulated_text)
        await self._queue.put(event)

        # Forward to next in chain if exists
        if self.next_in_chain is not None:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        """
        Called when the LLM generation is complete.
        Emits a TextCompleteEvent to the stream.
        """
        event = TextCompleteEvent(text=self._accumulated_text)
        self._queue.put_nowait(event)

        # Signal end of this generation
        self._queue.put_nowait(None)

        # Reset for next generation
        self._accumulated_text = ""

        # Forward to next in chain if exists
        if self.next_in_chain is not None:
            self.next_in_chain.flush()

    def close(self) -> None:
        """Close the stream. Call this when done consuming."""
        self._closed = True
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """
        Async iterator that yields streaming events.

        Yields TextDeltaEvent for each chunk and TextCompleteEvent when done.
        The iterator completes when flush() is called.
        """
        while not self._closed:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self.stream()

