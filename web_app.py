#!/usr/bin/env python3
"""
Kaine AI Web Interface - FastAPI Server
Provides a web interface for querying Telegram posts with rate limiting
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pythonjsonlogger import jsonlogger

try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from kaine_ai import TelegramQA

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Global instances (will be initialized in lifespan)
qa_tool = None
redis_client: Optional[Redis] = None
rate_limit_storage: Dict[str, deque] = None

# Setup structured JSON logging
def setup_logging():
    """Configure structured JSON logging for production"""
    log_handler = logging.StreamHandler(sys.stdout)

    if ENVIRONMENT == "production":
        # JSON formatter for production
        formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"}
        )
    else:
        # Simple formatter for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    log_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    logger.handlers.clear()
    logger.addHandler(log_handler)

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return logging.getLogger(__name__)

logger = setup_logging()

# Rate limiting configuration
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_MINUTES", "60")) * 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Replaces deprecated @app.on_event decorators
    """
    global qa_tool, redis_client, rate_limit_storage

    # Startup
    logger.info(f"Starting Kaine AI in {ENVIRONMENT} mode...")

    # Initialize rate limit storage
    rate_limit_storage = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX_REQUESTS))

    # Initialize Redis if available and configured
    if REDIS_AVAILABLE and REDIS_URL:
        try:
            redis_client = Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            redis_client.ping()
            logger.info(f"Redis connected: {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory rate limiting")
            redis_client = None
    else:
        logger.warning("Redis not available, using in-memory rate limiting")

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment!")
        logger.error("Please set it in your .env file")
        sys.exit(1)

    # Determine JSON file path
    if os.path.exists('data/mrkainez_posts.json'):
        json_file = 'data/mrkainez_posts.json'
    elif os.path.exists('data/sample_posts.json'):
        json_file = 'data/sample_posts.json'
    else:
        logger.error("No JSON data file found!")
        logger.error("Please ensure mrkainez_posts.json or sample_posts.json exists in data/ directory")
        sys.exit(1)

    try:
        qa_tool = TelegramQA(json_file, api_key)
        logger.info(f"Kaine AI initialized with {len(qa_tool.posts)} posts from {json_file}")
        logger.info(f"Rate limiting: {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes per IP")
        logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")
    except Exception as e:
        logger.error(f"ERROR initializing Kaine AI: {e}", exc_info=True)
        sys.exit(1)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Kaine AI...")

    if redis_client:
        try:
            redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(title="Kaine AI Web Interface", lifespan=lifespan)

# CORS middleware - restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict]


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    # Check X-Forwarded-For header first (for proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Fall back to direct client
    return request.client.host


def check_rate_limit_redis(ip_address: str) -> bool:
    """
    Redis-backed rate limiting using sliding window

    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    try:
        key = f"rate_limit:{ip_address}"
        now = datetime.now().timestamp()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        # Remove old entries
        redis_client.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        request_count = redis_client.zcard(key)

        if request_count >= RATE_LIMIT_MAX_REQUESTS:
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return False

        # Add current request
        redis_client.zadd(key, {str(now): now})

        # Set expiry on the key
        redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)

        return True

    except Exception as e:
        logger.error(f"Redis rate limit error: {e}, falling back to allow")
        # Fail open - allow request if Redis fails
        return True


def check_rate_limit_memory(ip_address: str) -> bool:
    """
    In-memory rate limiting (fallback when Redis unavailable)

    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    # Get request timestamps for this IP
    timestamps = rate_limit_storage[ip_address]

    # Remove old timestamps outside the window
    while timestamps and timestamps[0] < cutoff_time:
        timestamps.popleft()

    # Check if limit exceeded
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {ip_address}")
        return False

    # Add current request timestamp
    timestamps.append(now)

    return True


def check_rate_limit(ip_address: str) -> bool:
    """
    Check rate limit using Redis if available, otherwise fall back to in-memory

    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    if redis_client:
        return check_rate_limit_redis(ip_address)
    else:
        return check_rate_limit_memory(ip_address)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    index_path = Path("static/index.html")
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=500
        )

    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: Request, question_req: QuestionRequest):
    """
    Answer a question about Telegram posts

    Rate limited per IP address (configurable via environment)
    """
    # Get client IP
    client_ip = get_client_ip(request)

    logger.info(f"Question request from IP: {client_ip}")

    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"You have exceeded the limit of {RATE_LIMIT_MAX_REQUESTS} questions per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes. Please try again later.",
                "retry_after_minutes": RATE_LIMIT_WINDOW_SECONDS // 60
            }
        )

    # Validate question
    question = question_req.question.strip()
    if not question:
        logger.warning(f"Empty question from IP: {client_ip}")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > 1000:
        logger.warning(f"Question too long from IP: {client_ip}")
        raise HTTPException(status_code=400, detail="Question is too long (max 1000 characters)")

    try:
        logger.info(f"Processing question: {question[:100]}..." if len(question) > 100 else f"Processing question: {question}")

        # Get answer with sources
        result = qa_tool.answer_question(question, return_sources=True)

        logger.info(f"Successfully answered question from IP: {client_ip}")

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error answering question from IP {client_ip}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """
    Enhanced health check endpoint

    Checks:
    - QA tool initialization
    - Redis connectivity (if configured)
    - Basic system health
    """
    health_status = {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "posts_loaded": len(qa_tool.posts) if qa_tool else 0,
        "rate_limit": f"{RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes",
        "dependencies": {}
    }

    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            health_status["dependencies"]["redis"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
            logger.warning(f"Redis health check failed: {e}")
    else:
        health_status["dependencies"]["redis"] = "not_configured"

    # Check QA tool
    if not qa_tool:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["qa_tool"] = "not_initialized"
    else:
        health_status["dependencies"]["qa_tool"] = "healthy"

    # Return appropriate status code
    status_code = 200 if health_status["status"] in ["healthy", "degraded"] else 503

    return JSONResponse(content=health_status, status_code=status_code)


# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    """Run the web server"""
    # Determine number of workers based on environment
    workers = 1 if ENVIRONMENT == "development" else int(os.getenv("WORKERS", "4"))
    reload = ENVIRONMENT == "development"
    port = int(os.getenv("PORT", "8000"))

    logger.info("="*60)
    logger.info("Kaine AI Web Interface")
    logger.info("="*60)
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Port: {port}")
    logger.info("Starting server...")
    logger.info(f"Access the interface at: http://localhost:{port}")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*60)

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=reload,
        log_level=LOG_LEVEL.lower(),
        access_log=ENVIRONMENT == "development",
        timeout_keep_alive=75,
        limit_concurrency=1000,
        limit_max_requests=10000,
    )


if __name__ == "__main__":
    main()
