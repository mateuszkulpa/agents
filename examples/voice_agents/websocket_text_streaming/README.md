# WebSocket Text Streaming Example

This example demonstrates how to create a WebSocket server that streams LLM responses in real-time using a custom `StreamingTextOutput` class.

## Overview

The livekit-agents framework processes LLM responses internally with streaming, but the `RunResult` API only exposes complete events. This example shows how to tap into the real-time text stream using a custom `TextOutput` implementation.

## Components

- **`streaming_text_output.py`** - The `StreamingTextOutput` class that captures text deltas in real-time
- **`websocket_server.py`** - A WebSocket server that streams LLM responses to connected clients
- **`websocket_client.html`** - A browser-based client for testing the streaming
- **`test_streaming_text_output.py`** - Tests for the streaming functionality

## Installation

```bash
# Install dev dependencies (includes websockets)
uv sync --group dev

# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key
```

## Running the Example

### 1. Start the WebSocket Server

```bash
cd examples/voice_agents/websocket_text_streaming
uv run python websocket_server.py
```

You should see:
```
Starting WebSocket server on ws://localhost:8765
Connect using a WebSocket client or open websocket_client.html
```

### 2. Connect with the Web Client

Open `websocket_client.html` in your browser. The client will automatically connect to the server.

### 3. Send Messages

Type a message like "What is the weather in San Francisco?" and watch the response stream in real-time!

## How It Works

### StreamingTextOutput

The `StreamingTextOutput` class implements the `TextOutput` interface:

```python
from streaming_text_output import StreamingTextOutput

# Create the streaming output
streaming_output = StreamingTextOutput()

# Attach it to the session
sess.output.transcription = streaming_output

# Stream events as they happen
async for event in streaming_output:
    if event.type == "text_delta":
        print(f"Delta: {event.delta}")
    elif event.type == "text_complete":
        print(f"Complete: {event.text}")
```

### Event Types

- **`TextDeltaEvent`** - Emitted for each text chunk from the LLM
  - `delta`: The new text chunk
  - `accumulated`: All text accumulated so far

- **`TextCompleteEvent`** - Emitted when generation is complete
  - `text`: The complete generated text

## Using with WebSocket

```python
from streaming_text_output import StreamingTextOutput

async def handle_websocket(websocket):
    async with AgentSession(llm=llm) as sess:
        streaming_output = StreamingTextOutput()
        sess.output.transcription = streaming_output
        
        await sess.start(MyAgent())
        
        # Start the agent
        result = sess.run(user_input=user_message)
        
        # Stream to WebSocket
        async for event in streaming_output:
            if event.type == "text_delta":
                await websocket.send(json.dumps({
                    "type": "delta",
                    "text": event.delta
                }))
```

## Running Tests

```bash
# From the project root
uv run pytest examples/voice_agents/websocket_text_streaming/test_streaming_text_output.py -v
```

## Alternative Testing with websocat

If you have [websocat](https://github.com/vi/websocat) installed:

```bash
websocat ws://localhost:8765
```

Then type JSON messages:
```json
{"message": "What is the weather in San Francisco?"}
```

## Notes

- The `StreamingTextOutput` can be chained with other `TextOutput` implementations using the `next_in_chain` parameter
- The stream resets after each `flush()`, allowing multiple generations
- Use `close()` to stop the stream gracefully

