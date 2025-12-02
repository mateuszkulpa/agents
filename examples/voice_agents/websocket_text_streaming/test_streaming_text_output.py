"""
Tests for StreamingTextOutput

Run with: pytest examples/voice_agents/websocket_text_streaming/test_streaming_text_output.py -v
"""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai

from streaming_text_output import (
    StreamingTextOutput,
    TextCompleteEvent,
    TextDeltaEvent,
)


class TestAgent(Agent):
    """A simple test agent with a weather tool."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. Use the lookup_weather tool when asked about weather."
        )

    @function_tool
    async def lookup_weather(self, ctx: RunContext, location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city to get weather for
        """
        return "sunny with a temperature of 70 degrees."


@pytest.mark.asyncio
async def test_streaming_text_output_basic():
    """Test that StreamingTextOutput captures text deltas and completion."""
    streaming_output = StreamingTextOutput()

    # Simulate text capture
    await streaming_output.capture_text("Hello")
    await streaming_output.capture_text(" world")
    streaming_output.flush()

    # Collect events
    events = []
    async for event in streaming_output:
        events.append(event)

    # Verify events
    assert len(events) == 3

    assert isinstance(events[0], TextDeltaEvent)
    assert events[0].delta == "Hello"
    assert events[0].accumulated == "Hello"

    assert isinstance(events[1], TextDeltaEvent)
    assert events[1].delta == " world"
    assert events[1].accumulated == "Hello world"

    assert isinstance(events[2], TextCompleteEvent)
    assert events[2].text == "Hello world"


@pytest.mark.asyncio
async def test_streaming_text_output_accumulated():
    """Test accumulated_text property during streaming."""
    streaming_output = StreamingTextOutput()

    assert streaming_output.accumulated_text == ""

    await streaming_output.capture_text("First")
    assert streaming_output.accumulated_text == "First"

    await streaming_output.capture_text(" Second")
    assert streaming_output.accumulated_text == "First Second"

    streaming_output.flush()
    # After flush, accumulated text is reset
    assert streaming_output.accumulated_text == ""


@pytest.mark.asyncio
async def test_streaming_text_output_multiple_generations():
    """Test that StreamingTextOutput can handle multiple generations."""
    streaming_output = StreamingTextOutput()

    # First generation
    await streaming_output.capture_text("Gen 1")
    streaming_output.flush()

    events1 = []
    async for event in streaming_output:
        events1.append(event)

    assert len(events1) == 2
    assert events1[1].text == "Gen 1"

    # Second generation
    await streaming_output.capture_text("Gen 2")
    streaming_output.flush()

    events2 = []
    async for event in streaming_output:
        events2.append(event)

    assert len(events2) == 2
    assert events2[1].text == "Gen 2"


@pytest.mark.asyncio
async def test_streaming_text_output_with_chain():
    """Test that next_in_chain is called correctly."""
    collected_by_chain = []

    class MockTextOutput:
        async def capture_text(self, text: str) -> None:
            collected_by_chain.append(("capture", text))

        def flush(self) -> None:
            collected_by_chain.append(("flush",))

    mock_chain = MockTextOutput()
    streaming_output = StreamingTextOutput(next_in_chain=mock_chain)  # type: ignore

    await streaming_output.capture_text("Test")
    streaming_output.flush()

    assert ("capture", "Test") in collected_by_chain
    assert ("flush",) in collected_by_chain


@pytest.mark.asyncio
async def test_streaming_with_agent_session():
    """Integration test: StreamingTextOutput with a real AgentSession."""
    async with openai.LLM(model="gpt-4o-mini") as llm, AgentSession(llm=llm) as sess:
        streaming_output = StreamingTextOutput()
        sess.output.transcription = streaming_output

        await sess.start(TestAgent())

        # Start collecting events in background
        collected_events = []

        async def collect_events():
            async for event in streaming_output:
                collected_events.append(event)

        collector_task = asyncio.create_task(collect_events())

        # Run the agent with a weather query
        result = await sess.run(user_input="What is the weather in San Francisco?")

        # Wait for streaming to complete
        await asyncio.wait_for(collector_task, timeout=30.0)

        # Verify we got streaming events
        delta_events = [e for e in collected_events if isinstance(e, TextDeltaEvent)]
        complete_events = [e for e in collected_events if isinstance(e, TextCompleteEvent)]

        assert len(delta_events) > 0, "Expected at least one delta event"
        assert len(complete_events) == 1, "Expected exactly one complete event"

        # Verify the result also has expected events
        result.expect.next_event().is_function_call(
            name="lookup_weather", arguments={"location": "San Francisco"}
        )
        result.expect.next_event().is_function_call_output(
            output="sunny with a temperature of 70 degrees."
        )
        result.expect.next_event().is_message(role="assistant")


@pytest.mark.asyncio
async def test_streaming_events_are_real_time():
    """Test that events are emitted in real-time, not buffered."""
    streaming_output = StreamingTextOutput()

    received_times = []

    async def consumer():
        async for event in streaming_output:
            received_times.append(asyncio.get_event_loop().time())

    consumer_task = asyncio.create_task(consumer())

    # Send events with delays
    await streaming_output.capture_text("A")
    await asyncio.sleep(0.1)
    await streaming_output.capture_text("B")
    await asyncio.sleep(0.1)
    await streaming_output.capture_text("C")
    streaming_output.flush()

    await asyncio.wait_for(consumer_task, timeout=5.0)

    # Verify events were received at different times (real-time streaming)
    assert len(received_times) == 4  # 3 deltas + 1 complete
    assert received_times[1] - received_times[0] >= 0.09  # Allow small timing variance
    assert received_times[2] - received_times[1] >= 0.09


@pytest.mark.asyncio
async def test_close():
    """Test that close() stops the stream."""
    streaming_output = StreamingTextOutput()

    async def consumer():
        events = []
        async for event in streaming_output:
            events.append(event)
        return events

    consumer_task = asyncio.create_task(consumer())

    await streaming_output.capture_text("Before close")
    streaming_output.close()

    events = await asyncio.wait_for(consumer_task, timeout=1.0)

    assert len(events) == 1
    assert isinstance(events[0], TextDeltaEvent)
    assert events[0].delta == "Before close"

