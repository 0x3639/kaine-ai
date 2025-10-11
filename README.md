# Kaine AI - Telegram Q&A Tool

Kaine AI lets you ask natural language questions about Telegram chat history. It uses OpenAI's embeddings to semantically search through messages and GPT-4o to generate answers with relevant context.

For example, you can ask "What did they say about Bitcoin?" and get an AI-generated answer citing specific messages from your chat history.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/0x3639/kaine-ai.git
cd kaine-ai

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# Edit .env and replace "your-api-key-here" with your actual OpenAI API key

# 5. Run the tool (uses included sample data)
python kaine_ai.py
```

That's it! The tool will automatically generate embeddings on first run (takes a few minutes), then you can start asking questions.

## Troubleshooting

### "Error: OpenAI API key not found" or "Incorrect API key provided"
1. Make sure you created the `.env` file: `cp .env.example .env`
2. Open `.env` and replace `your-api-key-here` with your actual OpenAI API key
3. Get your API key from: https://platform.openai.com/api-keys

### "Can't find embeddings file" (First Run)
This is normal! The tool automatically generates embeddings on first run:
- Takes 1-2 minutes for 100 posts
- Takes 5-8 minutes for 500 posts
- Takes 10-15 minutes for 1000+ posts

Subsequent runs will be instant as embeddings are cached.

### "Error: File not found"
The default data file is `data/mrkainez_posts.json`. If you want to use a different file:
```bash
python kaine_ai.py data/your_file.json
```

### High API Costs
Each question costs approximately $0.02. To reduce costs:
1. Use fewer context posts (edit `DEFAULT_CONTEXT_POSTS=15` in `.env`)
2. Switch to cheaper models (edit `EMBEDDING_MODEL` and `CHAT_MODEL` in `.env`)

## Using Your Own Data

To use your own Telegram chat data:

1. Export your Telegram chat history as JSON
2. Place the file in the `data/` directory
3. Run: `python kaine_ai.py data/your_file.json`

Your JSON file should have this structure:
```json
{
  "posts": [
    {
      "message_id": 1,
      "date": "2024-01-15 10:30:00",
      "content": "Message text here",
      "user": {"id": 123456789, "username": "username"},
      "telegram_url": "https://t.me/channel/1"
    }
  ]
}
```

## How It Works

1. **Chunking**: Splits long messages into 512-token chunks
2. **Embeddings**: Converts text into vectors for semantic search
3. **Hybrid Search**: Combines semantic similarity with keyword matching
4. **Context Retrieval**: Finds the 15 most relevant message chunks
5. **Answer Generation**: GPT-4o generates an answer citing relevant messages

## Advanced Configuration

Edit `.env` to customize behavior:
- `EMBEDDING_MODEL`: text-embedding-3-large (default) or text-embedding-3-small (cheaper)
- `CHAT_MODEL`: gpt-4o (default) or gpt-4-turbo-preview
- `DEFAULT_CONTEXT_POSTS`: 15 (default) - number of relevant chunks to use
- `ENABLE_COST_TRACKING`: false (default) - set to true to see API costs per query
