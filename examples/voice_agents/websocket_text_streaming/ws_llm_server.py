"""
Standalone WebSocket LLM Server - Healthcare Assistant

This is a simple WebSocket server that acts as a healthcare-focused LLM assistant.
It uses OpenAI as the backend but streams responses over WebSocket.
This server is completely independent of LiveKit.

Protocol:
    Client -> Server: {"type": "chat", "messages": [{"role": "user", "content": "..."}]}
    Server -> Client: {"type": "delta", "content": "..."} (multiple times during streaming)
    Server -> Client: {"type": "complete", "content": "full response"}
    Server -> Client: {"type": "symptoms_collected", "symptoms": [...]}
    Server -> Client: {"type": "error", "message": "error description"}

Usage:
    uv run ws_llm_server.py

    The server will start on ws://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from websockets.asyncio.server import ServerConnection, serve

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws-llm-server")


class Symptom(BaseModel):
    """A single symptom with its status."""

    name: str = Field(
        description="Name of the symptom (e.g., 'headache', 'fever', 'cough')"
    )
    choice_id: Literal["present", "absent", "unknown"] = Field(
        description="Status of the symptom: 'present' if user has it, 'absent' if user confirmed they don't have it, 'unknown' if not discussed or unclear"
    )


class CollectSymptomsArgs(BaseModel):
    """Arguments for the collect_symptoms function."""

    symptoms: list[Symptom] = Field(
        description="List of symptoms collected from the user"
    )


def build_collect_symptoms_tool() -> dict[str, Any]:
    """Build the OpenAI function tool definition from Pydantic model."""
    return {
        "type": "function",
        "function": {
            "name": "collect_symptoms",
            "description": "Collect and record the symptoms reported by the user during the conversation. Call this function when the user has described their symptoms and you have gathered enough information.",
            "parameters": CollectSymptomsArgs.model_json_schema(),
        },
    }


COLLECT_SYMPTOMS_TOOL = build_collect_symptoms_tool()

# Healthcare-focused system prompt
HEALTHCARE_SYSTEM_PROMPT = """You are a helpful, empathetic healthcare assistant. Your role is to:

1. Provide general health information and wellness guidance
2. Help users understand medical terminology in simple terms
3. Encourage users to seek professional medical advice for specific conditions
4. Offer emotional support and active listening
5. Provide information about healthy lifestyle choices
6. Collect and record symptoms reported by users

Important guidelines:
- Always recommend consulting with a healthcare professional for specific medical advice
- Never diagnose conditions or prescribe treatments
- Be empathetic and understanding
- Use clear, simple language
- If someone describes an emergency, advise them to call emergency services immediately
- When the user describes symptoms, use the collect_symptoms function to record them
- Mark symptoms as 'present' if the user confirms they have them
- Mark symptoms as 'absent' if the user explicitly says they don't have them
- Mark symptoms as 'unknown' if not discussed or unclear

Remember: You are not a replacement for professional medical care. Your role is to inform,
support, and guide users toward appropriate healthcare resources."""


class HealthcareLLMServer:
    """WebSocket server that provides healthcare LLM assistance."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        openai_model: str = "gpt-4.1-mini",
    ):
        self.host = host
        self.port = port
        self.openai_model = openai_model
        self._client = openai.AsyncOpenAI()

    async def handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket connection."""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")

        # Maintain conversation history for this connection
        conversation_history: list[dict[str, Any]] = [
            {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT}
        ]

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data, conversation_history)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.exception(f"Error handling message from client {client_id}")
                    await self._send_error(websocket, str(e))
        except Exception as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            logger.info(f"Client {client_id} connection closed")

    async def _handle_message(
        self,
        websocket: ServerConnection,
        data: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Process incoming message and generate streaming response."""
        msg_type = data.get("type")

        if msg_type == "chat":
            messages = data.get("messages", [])
            if not messages:
                await self._send_error(websocket, "No messages provided")
                return

            # Add new messages to conversation history
            for msg in messages:
                if msg.get("role") and msg.get("content"):
                    conversation_history.append({"role": msg["role"], "content": msg["content"]})

            # Generate streaming response
            await self._generate_streaming_response(websocket, conversation_history)

        elif msg_type == "reset":
            # Reset conversation history (keep system prompt)
            conversation_history.clear()
            conversation_history.append({"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT})
            await websocket.send(
                json.dumps({"type": "reset_complete", "message": "Conversation history cleared"})
            )

        else:
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _generate_streaming_response(
        self,
        websocket: ServerConnection,
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Generate and stream LLM response."""
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        try:
            stream = await self._client.chat.completions.create(
                model=self.openai_model,
                messages=conversation_history,
                tools=[COLLECT_SYMPTOMS_TOOL],
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    full_response += delta.content
                    await websocket.send(json.dumps({"type": "delta", "content": delta.content}))

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        idx = tool_call.index
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tool_call.id or "",
                                "name": "",
                                "arguments": "",
                            }
                            logger.debug(f"New tool call at index {idx}")

                        if tool_call.id:
                            tool_calls[idx]["id"] = tool_call.id
                        if tool_call.function:
                            if tool_call.function.name:
                                tool_calls[idx]["name"] = tool_call.function.name
                                logger.debug(f"Tool call {idx} name: {tool_call.function.name}")
                            if tool_call.function.arguments:
                                logger.debug(f"Tool call {idx} args chunk: {tool_call.function.arguments!r}")
                                tool_calls[idx]["arguments"] += tool_call.function.arguments

            # Process any tool calls
            if tool_calls:
                await self._handle_tool_calls(websocket, tool_calls, conversation_history)

            # Send complete message (may be empty if only tool calls were made)
            await websocket.send(json.dumps({"type": "complete", "content": full_response}))

            # Add assistant response to conversation history
            if full_response:
                conversation_history.append({"role": "assistant", "content": full_response})

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            await self._send_error(websocket, f"LLM API error: {e.message}")

    async def _handle_tool_calls(
        self,
        websocket: ServerConnection,
        tool_calls: dict[int, dict[str, Any]],
        conversation_history: list[dict[str, Any]],
    ) -> None:
        """Handle tool calls from the LLM response."""
        for idx, tool_call in tool_calls.items():
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            raw_arguments = tool_call["arguments"]

            logger.debug(f"Tool call {idx}: name={tool_name}, id={tool_id}, args={raw_arguments}")

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {raw_arguments!r}, error: {e}")
                continue

            if tool_name == "collect_symptoms":
                try:
                    await self._handle_collect_symptoms(websocket, arguments)
                except Exception as e:
                    logger.error(f"Failed to handle collect_symptoms: {e}, arguments: {arguments}")
                    continue

                # Add tool call and result to conversation history
                conversation_history.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": raw_arguments,
                                },
                            }
                        ],
                    }
                )
                conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": "Symptoms collected successfully",
                    }
                )

    async def _handle_collect_symptoms(
        self,
        websocket: ServerConnection,
        arguments: dict[str, Any],
    ) -> None:
        """Handle the collect_symptoms function call and emit websocket event."""
        # Validate using Pydantic model
        validated_args = CollectSymptomsArgs.model_validate(arguments)

        logger.info(f"Symptoms collected: {validated_args.symptoms}")

        # Emit symptoms_collected event to client
        await websocket.send(
            json.dumps(
                {
                    "type": "symptoms_collected",
                    "symptoms": [s.model_dump() for s in validated_args.symptoms],
                }
            )
        )

    async def _send_error(self, websocket: ServerConnection, message: str) -> None:
        """Send error message to client."""
        await websocket.send(json.dumps({"type": "error", "message": message}))

    async def start(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting Healthcare LLM Server on ws://{self.host}:{self.port}")

        async with serve(self.handle_connection, self.host, self.port):
            logger.info("Server started. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        return

    server = HealthcareLLMServer(
        host="localhost",
        port=8765,
        openai_model="gpt-4.1-mini",
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
