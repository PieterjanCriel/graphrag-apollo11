# GraphRAG Apollo 11

A project demonstrating both basic RAG and Microsoft GraphRAG implementations using Apollo 11 mission transcripts with AWS Bedrock (Nova models).

## Overview

This project contains two RAG implementations:
1. **Basic RAG** (`basic-rag/`) - Simple vector-based retrieval using ChromaDB
2. **GraphRAG** (`apollo11/`) - [Microsoft's GraphRAG](https://microsoft.github.io/graphrag/) with knowledge graph extraction

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- AWS account with Bedrock access (Nova models enabled)
- AWS credentials configured (`~/.aws/credentials` or environment variables)

## Installation

Install all dependencies using `uv`:

```bash
uv sync
```

This will install:
- `graphrag>=2.7.0` - Microsoft GraphRAG framework
- `chromadb>=1.3.7` - Vector database for basic RAG
- `boto3>=1.42.15` - AWS SDK for Bedrock
- `python-dotenv>=1.2.1` - Environment variable management

## Configuration

### Environment Variables

The project uses two `.env` files for configuration:

#### 1. `basic-rag/.env` - Basic RAG Configuration

```env
# AWS Configuration
AWS_REGION=us-east-1

# ChromaDB Configuration
CHROMA_DIR=./chroma_apollo11
CHROMA_COLLECTION=apollo11_transcript

# Embedding Model Configuration
EMBEDDING_MODEL_ID=amazon.nova-2-multimodal-embeddings-v1:0
EMBEDDING_DIM=3072

# Chat Model Configuration
CHAT_MODEL_ID=us.amazon.nova-pro-v1:0

# RAG Configuration
TOP_K=6
MAX_CONTEXT_CHARS=12000
MAX_TOKENS=12000

# Processing Configuration
CHUNK_SIZE=1200
OVERLAP=100
BATCH_SIZE=16

# Transcript Path
TRANSCRIPT_PATH=./apollo11/input/transcription_apollo_11.txt
```

#### 2. `apollo11/.env` - GraphRAG Configuration

```env
AWS_REGION=us-east-1
AWS_BEDROCK_API_KEY=bedrock-api-key-fake
CHUNK_SIZE=1200
OVERLAP=100
EMBEDDING_MODEL_ID=amazon.nova-2-multimodal-embeddings-v1:0
CHAT_MODEL_ID=us.amazon.nova-pro-v1:0
```

**Note**: The `AWS_BEDROCK_API_KEY` is a placeholder required by GraphRAG's configuration but not actually used. AWS Bedrock authentication uses your AWS credentials from `~/.aws/credentials` or IAM roles.

### GraphRAG Settings

The `apollo11/settings.yaml` file configures GraphRAG to use AWS Bedrock:

- **Models**: Uses Nova Pro for chat and Nova Multimodal Embeddings
- **Input**: Reads from `apollo11/input/` directory
- **Output**: Stores results in `apollo11/output/`
- **Vector Store**: Uses LanceDB for embeddings
- **Chunking**: Configurable via environment variables

Environment variables are referenced using `${VARIABLE_NAME}` syntax in the YAML file.

---

## Basic RAG Usage

### 1. Process and Index Transcripts

Run the processing script to chunk the Apollo 11 transcript and create embeddings:

```bash
# Can be run from anywhere - paths are resolved automatically
uv run python basic-rag/process.py

# Or from within the basic-rag directory
cd basic-rag
uv run python process.py
```

This will:
- Read the transcript from `apollo11/input/transcription_apollo_11.txt`
- Chunk it into 1200-character segments with 100-character overlap
- Generate embeddings using AWS Bedrock Nova
- Store in ChromaDB at `basic-rag/chroma_apollo11`

**Note**: The script automatically resolves paths relative to its location, so it works from any directory.

### 2. Query the Basic RAG

Ask questions about the Apollo 11 mission:

```bash
# Can be run from anywhere
uv run python basic-rag/script.py --query "Which ship retrieved the astronauts?"

# Or from within the basic-rag directory
cd basic-rag
uv run python script.py --query "Which ship retrieved the astronauts?"
```

The script will:
- Embed your query using Nova embeddings
- Retrieve the top 6 most relevant chunks from ChromaDB
- Generate an answer using Nova Pro chat model
- **Log everything to `basic-rag/logs/query.log`**:
  - Timestamp
  - Your query
  - Retrieved chunks used for context
  - Generated answer

### 3. View Query History

All queries and their results are logged to `basic-rag/logs/query.log`:

```bash
cat basic-rag/logs/query.log
```

Each log entry includes:
- Timestamp of the query
- The question asked
- All retrieved chunks used for context
- The generated answer

---

## GraphRAG Usage

### 1. Initialize GraphRAG (Already Done)

The GraphRAG workspace was initialized with:

```bash
uv run graphrag init --root ./apollo11
```

This created:
- `apollo11/settings.yaml` - Configuration file
- `apollo11/prompts/` - Customizable prompt templates
- `apollo11/input/` - Directory for source documents

### 2. Index the Knowledge Graph

Build the knowledge graph from the Apollo 11 transcript:

```bash
uv run graphrag index --root ./apollo11
```

This process will:
- Extract entities (people, organizations, locations, events)
- Build relationships between entities
- Create community summaries
- Generate embeddings for semantic search
- Store results in `apollo11/output/`

**Note**: Indexing can take several minutes and will make multiple API calls to AWS Bedrock.

### 3. Query the Knowledge Graph

#### Local Search (Entity-focused)

Best for specific questions about entities and their relationships:

```bash
uv run graphrag query \
  --root ./apollo11 \
  --method local \
  --query "Which ship retrieved the astronauts?"
```

#### Global Search (Holistic understanding)

Best for broad questions requiring understanding of the entire dataset:

```bash
uv run graphrag query \
  --root ./apollo11 \
  --method global \
  --query "What were the major events during the Apollo 11 mission?"
```

#### Other Search Methods

- **drift**: For exploratory search across the knowledge graph
- **basic**: Simple semantic search without graph traversal


---

## AWS Bedrock Models Used

### Nova Pro (`us.amazon.nova-pro-v1:0`)
- **Purpose**: Chat/text generation
- **Use**: Answering queries, generating summaries
- **Max tokens**: 10,000

### Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`)
- **Purpose**: Text embeddings
- **Dimensions**: 3,072
- **Use**: Semantic search and retrieval
