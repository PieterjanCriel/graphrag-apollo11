import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from tqdm import tqdm

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load environment variables from .env file in the script's directory
load_dotenv(SCRIPT_DIR / ".env")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 100) -> List[str]:
    """
    Chunk `text` into fixed-size character chunks with overlap.
    Example: chunk_size=1200, overlap=100 => step=1100.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def bedrock_nova_embed_texts(
    texts: List[str],
    *,
    region_name: str = "us-east-1",
    model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
    embedding_dimension: int = 3072,
    max_retries: int = 3,
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Embed a list of texts using Bedrock Runtime + Nova embeddings via SINGLE_EMBEDDING.
    (Calls Bedrock once per text with retry logic.)

    Args:
        texts: List of text strings to embed
        region_name: AWS region
        model_id: Bedrock model ID
        embedding_dimension: Embedding dimension
        max_retries: Maximum number of retries for failed API calls
        show_progress: Whether to show progress bar

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)

    embeddings: List[List[float]] = []
    iterator = tqdm(texts, desc="Embedding texts", disable=not show_progress)

    for t in iterator:
        for attempt in range(max_retries):
            try:
                request_body = {
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX",
                        "embeddingDimension": embedding_dimension,
                        "text": {"truncationMode": "END", "value": t},
                    },
                }

                response = bedrock_runtime.invoke_model(
                    body=json.dumps(request_body),
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = json.loads(response["body"].read())

                # Nova response shape can vary by SDK/model version.
                # These fallbacks try common patterns.
                emb = (
                    response_body.get("embedding")
                    or response_body.get("embeddings")
                    or response_body.get("output", {}).get("embedding")
                    or response_body.get("output", {}).get("embeddings")
                )

                if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                    embeddings.append(emb)
                    break  # Success, exit retry loop
                elif isinstance(emb, list) and emb and isinstance(emb[0], dict) and "embedding" in emb[0]:
                    embeddings.append(emb[0]["embedding"])
                    break  # Success, exit retry loop
                else:
                    raise RuntimeError(f"Unexpected embedding response format: {response_body}")

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')

                if error_code == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                        if show_progress:
                            iterator.write(f"⚠️  Rate limited. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Max retries ({max_retries}) exceeded due to throttling") from e
                elif error_code in ['ModelTimeoutException', 'ServiceUnavailableException']:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        if show_progress:
                            iterator.write(f"⚠️  Service error ({error_code}). Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Max retries ({max_retries}) exceeded due to {error_code}") from e
                else:
                    # For other errors, don't retry
                    raise
            except Exception as e:
                # For non-ClientError exceptions, retry with backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    if show_progress:
                        iterator.write(f"⚠️  Error: {str(e)}. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

    return embeddings


def embed_transcript_to_chromadb(
    transcript_path: str = "transcription_apollo_11.txt",
    *,
    chunk_size: int = 1200,
    overlap: int = 100,
    persist_dir: str = "./chroma_apollo11",
    collection_name: str = "apollo11_transcript",
    region_name: str = "us-east-1",
    model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
    embedding_dimension: int = 3072,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    1) Read transcript file
    2) Chunk into 1200 chars with 100 overlap (configurable)
    3) Embed chunks via Bedrock Nova embeddings
    4) Store documents+embeddings+metadata in local persistent ChromaDB
    """
    # Convert to Path and resolve relative to script directory if not absolute
    transcript_path_obj = Path(transcript_path)
    if not transcript_path_obj.is_absolute():
        # Try relative to script directory first
        transcript_path_obj = SCRIPT_DIR / transcript_path
        if not transcript_path_obj.exists():
            # Try relative to current working directory
            transcript_path_obj = Path(transcript_path).resolve()

    if not transcript_path_obj.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    # Resolve persist_dir relative to script directory if not absolute
    persist_dir_obj = Path(persist_dir)
    if not persist_dir_obj.is_absolute():
        persist_dir_obj = SCRIPT_DIR / persist_dir

    transcript_path = str(transcript_path_obj)
    persist_dir = str(persist_dir_obj)

    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📄 Reading transcript: {transcript_path}")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"✂️  Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    # Persistent local ChromaDB
    print(f"💾 Connecting to ChromaDB at: {persist_dir}")
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=collection_name)

    total_added = 0
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"🚀 Processing {num_batches} batches (batch_size={batch_size})")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches", unit="batch"):
        batch = chunks[i : i + batch_size]
        embs = bedrock_nova_embed_texts(
            batch,
            region_name=region_name,
            model_id=model_id,
            embedding_dimension=embedding_dimension,
            show_progress=False,  # Don't show nested progress bar
        )

        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [
            {
                "source_file": os.path.basename(transcript_path),
                "chunk_index": i + j,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "char_start": (i + j) * (chunk_size - overlap),
                "char_end": (i + j) * (chunk_size - overlap) + len(batch[j]),
                "model_id": model_id,
                "embedding_dimension": embedding_dimension,
            }
            for j in range(len(batch))
        ]

        collection.add(
            ids=ids,
            documents=batch,
            embeddings=embs,
            metadatas=metadatas,
        )
        total_added += len(batch)

    print(f"✅ Successfully added {total_added} chunks to ChromaDB")

    return {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "source_file": transcript_path,
        "chunks_created": len(chunks),
        "chunks_added": total_added,
    }


# Example usage:
if __name__ == "__main__":
    result = embed_transcript_to_chromadb(
        transcript_path=os.getenv("TRANSCRIPT_PATH", "./apollo11/transcription_apollo_11.txt"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        overlap=int(os.getenv("OVERLAP", "100")),
        persist_dir=os.getenv("CHROMA_DIR", "./chroma_apollo11"),
        collection_name=os.getenv("CHROMA_COLLECTION", "apollo11_transcript"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_id=os.getenv("EMBEDDING_MODEL_ID", "amazon.nova-2-multimodal-embeddings-v1:0"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIM", "3072")),
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
    )
    print(result)