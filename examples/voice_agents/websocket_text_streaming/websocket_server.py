"""
WebSocket Text Streaming Example

This example demonstrates how to create a WebSocket server that streams
LLM responses in real-time using the StreamingTextOutput class.

Run this server and connect to ws://localhost:8765 to test text streaming.

Example client (using websocat):
    websocat ws://localhost:8765
    > {"message": "What is the weather in San Francisco?"}

Or use the included HTML client by opening websocket_client.html in a browser.
"""

import asyncio
import json
import logging

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai

from streaming_text_output import StreamingTextOutput
from dotenv import load_dotenv

load_dotenv()

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Please install websockets: uv add websockets")
    raise


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAgent(Agent):
    """A simple agent with a weather lookup tool for demonstration."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. When asked about weather, "
            "use the lookup_weather tool to get the current conditions. "
            "Always provide a friendly, conversational response."
        )

    @function_tool
    async def lookup_weather(self, ctx: RunContext, location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        # Simulated weather data
        weather_data = {
            "san francisco": "sunny with a temperature of 68°F",
            "new york": "partly cloudy with a temperature of 72°F",
            "london": "rainy with a temperature of 55°F",
            "tokyo": "clear skies with a temperature of 75°F",
        }

        location_lower = location.lower()
        for city, weather in weather_data.items():
            if city in location_lower:
                return weather

        return f"Currently showing mild weather with a temperature of 65°F for {location}"


async def handle_websocket(websocket):
    """Handle a single WebSocket connection."""
    logger.info("New WebSocket connection established")

    try:
        async with openai.LLM(model="gpt-4o-mini") as llm:
            async with AgentSession(llm=llm) as sess:
                # Create the streaming output
                streaming_output = StreamingTextOutput()
                sess.output.transcription = streaming_output

                # Start the agent
                await sess.start(WeatherAgent())

                # Send ready message
                await websocket.send(
                    json.dumps({"type": "ready", "message": "Agent ready. Send a message to start."})
                )

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        user_message = data.get("message", "")

                        if not user_message:
                            await websocket.send(
                                json.dumps({"type": "error", "message": "No message provided"})
                            )
                            continue

                        logger.info(f"Received message: {user_message}")

                        # Start the agent run
                        result = sess.run(user_input=user_message)

                        # Stream the response
                        async for event in streaming_output:
                            if event.type == "text_delta":
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "delta",
                                            "delta": event.delta,
                                            "accumulated": event.accumulated,
                                        }
                                    )
                                )
                            elif event.type == "text_complete":
                                await websocket.send(
                                    json.dumps({"type": "complete", "text": event.text})
                                )

                        # Wait for the full result (includes tool calls etc.)
                        await result

                        # Send the events summary
                        events_summary = []
                        for ev in result.events:
                            if ev.type == "function_call":
                                events_summary.append(
                                    {
                                        "type": "function_call",
                                        "name": ev.item.name,
                                        "arguments": ev.item.arguments,
                                    }
                                )
                            elif ev.type == "function_call_output":
                                events_summary.append(
                                    {
                                        "type": "function_call_output",
                                        "output": ev.item.output,
                                    }
                                )
                            elif ev.type == "message":
                                events_summary.append(
                                    {
                                        "type": "message",
                                        "role": ev.item.role,
                                        "content": ev.item.text_content,
                                    }
                                )

                        await websocket.send(
                            json.dumps({"type": "result", "events": events_summary})
                        )

                    except json.JSONDecodeError:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "Invalid JSON"})
                        )

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.exception(f"Error in WebSocket handler: {e}")
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def main():
    """Start the WebSocket server."""
    host = "localhost"
    port = 8765

    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    logger.info("Connect using a WebSocket client or open websocket_client.html")

    async with serve(handle_websocket, host, port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())

