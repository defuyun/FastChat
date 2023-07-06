"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server
"""
import argparse
import asyncio
import json
import logging
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi import Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_socketio import SocketManager
import httpx
from pydantic import BaseSettings
import uvicorn

from fastchat.constants import (
    WORKER_API_TIMEOUT,
    ErrorCode,
)
from fastchat.conversation import Conversation, SeparatorStyle
from fastapi.exceptions import RequestValidationError
from fastchat.protocol.openai_api_protocol import (
    ErrorResponse
)
from fastchat.protocol.api_protocol import (
    SimpleChatCompletionRequest,
    SimpleCompletionResponse,
)

logger = logging.getLogger(__name__)

conv_template_map = {}


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    worker_address: str = "http://localhost:21002"
    model: str = None
    history: Dict[str, List[Dict[str, str]]] = {}

app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {}

def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

async def get_gen_params(
    model_name: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    top_k: float,
    max_tokens: Optional[int],
    echo: Optional[bool]
) -> Dict[str, Any]:
    conv = await get_conv(model_name)
    conv = Conversation(
        name=conv["name"],
        system=conv["system"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo
    }

    gen_params.update({"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def get_conv(model_name: str):
    async with httpx.AsyncClient() as client:
        worker_addr = app_settings.worker_address
        conv_template = conv_template_map.get((worker_addr, model_name))
        if conv_template is None:
            response = await client.post(
                worker_addr + "/worker_get_conv_template",
                headers=headers,
                json={},
                timeout=WORKER_API_TIMEOUT,
            )
            conv_template = response.json()["conv"]
            conv_template_map[(worker_addr, model_name)] = conv_template
        return conv_template


async def generate_completion(payload: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = app_settings.worker_address

        response = await client.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        completion = response.json()
        return completion

### GENERAL API - NOT OPENAI COMPATIBLE ###

@app.post("/api/chat")
async def create_chat_completion(request: SimpleChatCompletionRequest):
    """Creates a completion for the chat message"""
    model = app_settings.model
    session_id = request.session_id

    if not session_id in app_settings.history:
        app_settings.history[session_id] = []

    if session_id != "" and session_id != None:
        # assuming a user makes sequential requests for now...
        app_settings.history[session_id].append({"role": "user", "content": request.message})

    gen_params = await get_gen_params(
        model,
        app_settings.history[session_id] if session_id != "" and session_id != None else request.message,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_new_tokens,
        echo=False
    )

    response = await generate_completion(gen_params)

    if response["error_code"] != 0:
        return create_error_response(response["error_code"], response["text"])

    if session_id != "" and session_id != None:
        app_settings.history[session_id].append({"role": "assistant", "content": response["text"]})

    return SimpleCompletionResponse(message=response["text"])

### END GENERAL API - NOT OPENAI COMPATIBLE ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--model", type=str, default="falcon-40b-instruct"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.model = args.model

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
