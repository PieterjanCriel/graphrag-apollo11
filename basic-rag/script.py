#!/usr/bin/env python3
import os
import json
import argparse
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load environment variables from .env file in the script's directory
load_dotenv(SCRIPT_DIR / ".env")

# -----------------------------
# CONFIG (loaded from .env with defaults)
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Resolve CHROMA_DIR relative to script directory
CHROMA_DIR_RAW = os.getenv("CHROMA_DIR", "./chroma_apollo11")
CHROMA_DIR_PATH = Path(CHROMA_DIR_RAW)
if not CHROMA_DIR_PATH.is_absolute():
    CHROMA_DIR_PATH = SCRIPT_DIR / CHROMA_DIR_RAW
CHROMA_DIR = str(CHROMA_DIR_PATH)

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "apollo11_transcript")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.nova-2-multimodal-embeddings-v1:0")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "us.amazon.nova-pro-v1:0")

TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))

# Logs directory
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
QUERY_LOG_FILE = LOGS_DIR / "query.log"


# -----------------------------
# Embedding
# -----------------------------
def embed_query(text: str) -> list[float]:
    brt = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": EMBEDDING_DIM,
            "text": {"truncationMode": "END", "value": text},
        },
    }

    resp = brt.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    out = json.loads(resp["body"].read())
    emb = out.get("embedding") or out.get("embeddings")
    if isinstance(emb, list) and isinstance(emb[0], dict):
        emb = emb[0]["embedding"]

    return emb


# -----------------------------
# Retrieve from Chroma
# -----------------------------
def retrieve_chunks(query: str) -> list[str]:
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(CHROMA_COLLECTION)

    q_emb = embed_query(query)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents"],
    )

    return (res.get("documents") or [[]])[0]


def build_context(chunks: list[str], query: str) -> str:
    ctx = []
    used = 0

    for i, c in enumerate(chunks, 1):
        block = f"[Chunk {i}]\n{c.strip()}\n"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        ctx.append(block)
        used += len(block)

    context_text = "\n".join(ctx)

    # Log query and chunks to query.log
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_separator = "=" * 80

    with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{log_separator}\n")
        f.write(f"Timestamp: {now}\n")
        f.write(f"Query: {query}\n")
        f.write(f"{log_separator}\n\n")
        f.write("Retrieved Chunks:\n\n")
        f.write(context_text)
        f.write(f"\n\n{log_separator}\n\n")

    return context_text


# -----------------------------
# Converse (Nova Pro)
# -----------------------------
def chat_with_nova(query: str, context: str) -> str:
    brt = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    response = brt.converse(
        modelId=CHAT_MODEL_ID,
        system=[
            {
                "text": (
                    "You are a helpful assistant. "
                    "If the excerpts do not contain the answer, say so."
                    "You are not allowed to use any information outside the provided context and chunks or make up answers."
                    "below your answer, create the list of the sources used and only those."
                )
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "TRANSCRIPT EXCERPTS:\n"
                            f"{context if context else '[None]'}"
                        )
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"text": query}],
            },
        ],
        inferenceConfig={
            "maxTokens": MAX_TOKENS,
            "temperature": 0.1,
            "topP": 0.9,
        },
    )

    blocks = response.get("output", {}).get("message", {}).get("content", [])
    for b in blocks:
        if "text" in b:
            return b["text"]

    return ""


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Apollo 11 chat (ChromaDB + Nova Pro)")
    parser.add_argument("--query", required=True, help="Question to ask")
    args = parser.parse_args()

    chunks = retrieve_chunks(args.query)
    context = build_context(chunks, args.query)
    answer = chat_with_nova(args.query, context)

    # Log the answer as well
    with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Answer:\n{answer}\n")
        f.write(f"\n{'=' * 80}\n\n")

    print(answer)


if __name__ == "__main__":
    main()