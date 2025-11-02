# Kaine AI - Web Interface

A web-based interface for querying Telegram posts using AI-powered semantic search. Built with FastAPI and featuring a ChatGPT-style dark mode interface.

## Features

- **ChatGPT-Style Interface**: Clean, modern dark mode UI
- **Rate Limiting**: 10 questions per hour per IP address
- **Source Citations**: Each answer includes clickable Telegram links to source posts
- **Conversation History**: Tracks questions and answers during your session
- **Copy Responses**: Easy-to-use copy button for each answer
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Telegram posts data file (JSON format)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements-web.txt
   ```

2. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Ensure data file exists:**

   Place your Telegram posts JSON file in the `data/` directory:
   - `data/mrkainez_posts.json` (checked first)
   - `data/sample_posts.json` (fallback)

### Running the Server

**Start the web server:**
```bash
python web_app.py
```

**Or using uvicorn directly:**
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

**Access the interface:**
Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. Type your question in the text box at the bottom
2. Press Enter or click the send button (↑)
3. Wait for the AI to process your question (loading spinner will appear)
4. View the answer with source citations below
5. Click "View on Telegram →" to see the original posts
6. Use the Copy button to copy responses to your clipboard

## Rate Limiting

- **Limit**: 10 questions per hour per IP address
- **Window**: Rolling 60-minute window
- **Response**: HTTP 429 error with retry-after information when limit is exceeded
- **Storage**: In-memory (resets when server restarts)

## API Endpoints

### `GET /`
Serves the main web interface (HTML page)

### `POST /api/ask`
Submit a question and get an answer with sources

**Request Body:**
```json
{
  "question": "Your question here"
}
```

**Response:**
```json
{
  "answer": "AI-generated answer text",
  "sources": [
    {
      "date": "2024-01-15 10:30:00",
      "author": "John Doe (@johndoe)",
      "url": "https://t.me/channel/123",
      "relevance_score": 0.95
    }
  ]
}
```

**Error Responses:**
- `400`: Invalid request (empty question, too long)
- `429`: Rate limit exceeded
- `500`: Server error

### `GET /api/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "posts_loaded": 1234,
  "rate_limit": "10 per 60 minutes"
}
```

## Configuration

All configuration is done via environment variables in `.env`:

### Required
- `OPENAI_API_KEY` - Your OpenAI API key

### Optional (with defaults)
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHAT_MODEL` - Default: `gpt-4o`
- `DEFAULT_CONTEXT_POSTS` - Default: `15`
- `CHUNK_SIZE` - Default: `512`
- `CHUNK_OVERLAP` - Default: `50`
- `ENABLE_COMPRESSION` - Default: `false`
- `ENABLE_DIVERSITY` - Default: `true`
- `ENABLE_COST_TRACKING` - Default: `false`

## Project Structure

```
kaine-ai/
├── web_app.py              # FastAPI server with rate limiting
├── kaine_ai.py             # Core QA engine (modified to return sources)
├── requirements-web.txt    # Python dependencies
├── README-WEB.md          # This file
├── static/                # Frontend files
│   ├── index.html         # Main web page
│   ├── style.css          # Dark mode styling
│   └── script.js          # Frontend logic
├── data/                  # Input data
│   └── mrkainez_posts.json
├── cache/                 # Generated embeddings & indexes
└── .env                   # Environment configuration
```

## Deployment

### Local Development
```bash
python web_app.py
```
Server runs with auto-reload enabled for development.

### Production

**Using uvicorn:**
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using gunicorn with uvicorn workers:**
```bash
gunicorn web_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t kaine-ai-web .
docker run -p 8000:8000 --env-file .env kaine-ai-web
```

## Security Considerations

- **Rate Limiting**: Prevents abuse with IP-based rate limiting
- **Input Validation**: Questions are limited to 1000 characters
- **CORS**: Configured for local development (adjust for production)
- **No Authentication**: This is a simple demo - add authentication for production use
- **API Key Security**: Never commit `.env` file to git

## Troubleshooting

### Server won't start
- Check that `OPENAI_API_KEY` is set in `.env`
- Ensure data file exists in `data/` directory
- Verify all dependencies are installed: `pip install -r requirements-web.txt`

### Rate limit errors
- Wait for the 60-minute window to reset
- Restart the server to clear in-memory rate limits (development only)

### No sources showing
- Check that Telegram posts have `telegram_url` field
- Verify posts have `user` and `date` metadata

### Slow responses
- First query is slower (loads embeddings from cache)
- Subsequent queries are faster
- Consider enabling compression for very large contexts (increases cost)

## Cost Estimates

Per query (without compression):
- Embedding: ~$0.001
- Chat completion: ~$0.02
- **Total: ~$0.021 per query**

With compression enabled (not recommended):
- Additional: ~$0.20 per query
- **Total: ~$0.22 per query**

At 10 queries per user per hour, typical daily costs for a few users are minimal.

## Differences from CLI Version

The web interface (`web_app.py`) shares the same core engine as the CLI (`kaine_ai.py`) but adds:

1. **Web Interface**: HTML/CSS/JS frontend
2. **Rate Limiting**: IP-based request throttling
3. **API Endpoints**: RESTful API for questions
4. **Source Citations**: Returns structured source data
5. **Session Management**: Conversation history in frontend

The CLI version remains available on the `main` branch for interactive terminal use.

## License

Same as the main Kaine AI project.

## Support

For issues specific to the web interface:
1. Check this README
2. Review browser console for frontend errors
3. Check server logs for backend errors
4. Verify `.env` configuration

For core functionality issues, refer to the main `README.md` and `CLAUDE.md`.
