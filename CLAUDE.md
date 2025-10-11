# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaine AI is a Telegram Q&A analysis tool that uses OpenAI's embeddings and GPT-4o to perform semantic search over Telegram chat history. The tool implements hybrid search (semantic + keyword-based TF-IDF) with post chunking and optional context compression.

## Directory Structure

```
mr-kaine-ai/
├── data/                # Input JSON data files
│   └── mrkainez_posts.json
├── cache/               # Generated .pkl files (gitignored)
│   ├── .gitkeep
│   ├── *_embeddings.pkl
│   ├── *_chunks.pkl
│   └── *_tfidf.pkl
├── kaine_ai.py          # Main application
├── requirements.txt     # Python dependencies
├── README.md           # User documentation
├── CLAUDE.md           # Developer documentation
├── .env                # Local secrets (gitignored)
├── .env.example        # Environment template
└── .gitignore          # Git ignore rules
```

## Quick Commands

### Setup
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment (required)
cp .env.example .env
# Then edit .env and add your OpenAI API key
```

### Running the Tool
```bash
# Activate venv first (if using virtual environment)
source venv/bin/activate

# Run with default file (looks for data/sample_posts.json)
python kaine_ai.py

# Run with specific JSON file
python kaine_ai.py data/mrkainez_posts.json
```

### Interactive Commands
Once running, the tool provides an interactive chat interface:
- Type any question to query the Telegram posts
- Type `stats` to see statistics about loaded posts
- Type `quit`, `exit`, or `q` to exit

## Architecture

### Core Component: TelegramQA Class (kaine_ai.py)

The main class handles the entire lifecycle from data loading to question answering:

1. **Data Loading & Chunking** (lines 83-171)
   - Loads posts from JSON into memory
   - Splits long posts into chunks (configurable size, default 512 tokens with 50 token overlap)
   - Creates normalized searchable text for each post/chunk
   - Chunks are cached in `cache/*.pkl` files for performance

2. **Embedding System** (lines 173-217)
   - Creates embeddings for all chunks using OpenAI's text-embedding-3-large model
   - Processes in batches of 100 for efficiency
   - Normalizes vectors for cosine similarity
   - Embeddings are cached as `cache/*_embeddings.pkl` files
   - Cache directory is auto-created on first run (lines 50-52)

3. **Hybrid Search Architecture** (lines 239-376)
   - **Semantic Search**: Uses cosine similarity between query and chunk embeddings
   - **Keyword Search**: Implements custom TF-IDF scoring (lines 239-298)
   - **Score Fusion**: Combines semantic and keyword scores (default: 70% semantic, 30% keyword)
   - **Diversity Re-ranking**: Limits chunks from same post (max 3) to ensure diverse results

4. **Context Building & Compression** (lines 378-440)
   - Groups chunks back into posts for coherent context
   - Optional compression using GPT-4o for large contexts (disabled by default)
   - Compression threshold: 100,000 tokens (configurable)
   - Cost tracking available for monitoring API usage

5. **Answer Generation** (lines 442-489)
   - Takes user question and finds relevant chunks
   - Builds context from top results
   - Uses GPT-4o with system prompt specialized for Telegram post analysis
   - Includes dates, authors, and Telegram URLs in responses

### File Naming Convention

All generated files are stored in the `cache/` directory:
- `cache/{source}_embeddings.pkl` - Pre-computed embeddings
- `cache/{source}_chunks.pkl` - Post chunks with metadata
- `cache/{source}_tfidf.pkl` - TF-IDF vectors for keyword search

The cache directory is automatically created on first run and is excluded from git via .gitignore.

### Configuration via Environment Variables

All configurable parameters are in `.env`:

**Required:**
- `OPENAI_API_KEY` - OpenAI API key

**Model Selection:**
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHAT_MODEL` - Default: `gpt-4o`

**Search Settings:**
- `DEFAULT_CONTEXT_POSTS` - Default: 15 posts
- `CHUNK_SIZE` - Default: 512 tokens
- `CHUNK_OVERLAP` - Default: 50 tokens

**Feature Flags:**
- `ENABLE_COMPRESSION` - Default: false (expensive)
- `COMPRESSION_THRESHOLD` - Default: 100000 tokens
- `ENABLE_DIVERSITY` - Default: true
- `ENABLE_COST_TRACKING` - Default: false

### Data Format

Input JSON must follow this structure:
```json
{
  "posts": [
    {
      "message_id": int,
      "date": "YYYY-MM-DD HH:MM:SS",
      "content": "string",
      "user": {
        "id": int,
        "username": "string",
        "first_name": "string"
      },
      "telegram_url": "https://t.me/...",
      "media": null | object
    }
  ]
}
```

### Cost Considerations

The tool tracks API costs (when enabled):
- **Per query without compression**: ~$0.02
- **Per query with compression**: ~$0.22 (compression adds ~$0.20)
- **Initial embedding creation**: One-time cost based on corpus size

Pricing reference (as of code):
- gpt-4o: $2.50 input / $10.00 output per 1M tokens
- text-embedding-3-large: $0.13 per 1M tokens

## Development Notes

### Adding Features

When modifying the search algorithm:
- Semantic search logic: `find_relevant_chunks()` (line 300)
- TF-IDF implementation: `create_tfidf()` (line 253)
- Score fusion: Line 332-335
- Diversity re-ranking: `_diversity_rerank()` (line 353)

### Testing Changes

To test without incurring embedding costs:
1. Keep existing `cache/*.pkl` files intact
2. Modify only the query/answer logic
3. Delete `cache/*.pkl` files only when changing chunking/embedding logic
4. The `cache/` directory and its contents are gitignored, so cache files won't be committed

### Debugging

Enable cost tracking to monitor API usage:
```bash
ENABLE_COST_TRACKING=true python kaine_ai.py
```

This shows per-call costs and cumulative totals during execution.
