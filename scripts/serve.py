#!/usr/bin/env python3
"""OpenAI-compatible HTTP serving server for GemmaInfer on RTX 3090 / 4090.

Endpoints:
  - POST /v1/chat/completions (supports stream=True SSE and stream=False)
  - POST /v1/completions
  - GET  /v1/models
  - GET  /health

Usage:
  python scripts/serve.py --host 0.0.0.0 --port 8000 --synthetic
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import List, Optional, Union, Dict, Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.engine import GemmaInferEngine
from gemmainfer.utils.weights import load_gemma_from_hf

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(engine: GemmaInferEngine, model_name: str = "gemma-4-26b-a4b", tokenizer=None):
    if not HAS_FASTAPI:
        raise ImportError("FastAPI and Uvicorn required for serving. Run: pip install fastapi uvicorn")

    app = FastAPI(title="GemmaInfer Serving Engine", version="0.1.0")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: Optional[str] = model_name
        messages: List[ChatMessage]
        max_tokens: Optional[int] = 128
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.95
        stream: Optional[bool] = False

    class CompletionRequest(BaseModel):
        model: Optional[str] = model_name
        prompt: Union[str, List[int]]
        max_tokens: Optional[int] = 128
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.95
        stream: Optional[bool] = False

    def tokenize_messages(messages: List[ChatMessage]) -> List[int]:
        prompt_text = ""
        for m in messages:
            prompt_text += f"<start_of_turn>{m.role}\n{m.content}<end_of_turn>\n"
        prompt_text += "<start_of_turn>model\n"

        if tokenizer is not None:
            return tokenizer.encode(prompt_text)
        return [ord(c) % 256000 for c in prompt_text]

    def decode_token(tok: int) -> str:
        if tokenizer is not None:
            return tokenizer.decode([tok])
        return chr(tok % 128) if 32 <= (tok % 128) <= 126 else " "

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "device": str(engine.device),
            "vram_allocated_gb": (
                round(torch.cuda.memory_allocated() / (1024**3), 2)
                if torch.cuda.is_available()
                else 0.0
            ),
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "gemmainfer",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        prompt_ids = tokenize_messages(req.messages)
        if not prompt_ids:
            prompt_ids = [1]

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if req.stream:
            async def event_generator():
                for tok in engine.generate_stream(
                    prompt_token_ids=prompt_ids,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                ):
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": decode_token(tok)},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming response
        tokens = engine.generate(
            prompt_token_ids=prompt_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        content = "".join([decode_token(t) for t in tokens])

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt_ids) + len(tokens),
            },
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="GemmaInfer Serving Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    if not HAS_FASTAPI:
        print("[ERROR] FastAPI and Uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    dtype = getattr(torch, args.dtype)

    if args.model_path and not args.synthetic:
        print(f"Loading weights from {args.model_path}...")
        model, config = load_gemma_from_hf(args.model_path, device=torch.device(args.device), dtype=dtype)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except Exception:
            tokenizer = None
    else:
        print("Initializing synthetic Gemma 4 26B A4B model...")
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        model = GemmaForCausalLM(config).to(device=args.device, dtype=dtype)
        tokenizer = None

    engine = GemmaInferEngine(
        config=config,
        model=model,
        device=args.device,
        dtype=dtype,
        enable_cuda_graphs=True,
        enable_dspark=True,
    )

    app = create_app(engine, tokenizer=tokenizer)
    print(f"\nGemmaInfer Server listening on http://{args.host}:{args.port}")
    print(f"OpenAI Chat API: http://{args.host}:{args.port}/v1/chat/completions\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
