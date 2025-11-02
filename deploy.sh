#!/bin/bash
#
# Kaine AI Production Deployment Script
# This script automates the deployment process for production environments
#
# Usage:
#   ./deploy.sh [--no-backup] [--rebuild]
#
# Options:
#   --no-backup   Skip cache backup before deployment
#   --rebuild     Force rebuild of Docker images (no cache)
#   --help        Show this help message

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
BACKUP=true
REBUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-backup)
            BACKUP=false
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --help)
            head -n 12 "$0" | tail -n 10
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Kaine AI Production Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running in correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file from .env.example:"
    echo "  cp .env.example .env"
    echo "  nano .env  # Edit with your configuration"
    exit 1
fi

# Check if required environment variables are set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-api-key-here" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not configured in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment configuration validated"

# Backup cache directory if it exists
if [ "$BACKUP" = true ] && [ -d "cache" ] && [ "$(ls -A cache)" ]; then
    BACKUP_DIR="cache_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}→${NC} Backing up cache directory to $BACKUP_DIR..."
    cp -r cache "$BACKUP_DIR"
    echo -e "${GREEN}✓${NC} Cache backed up"
else
    echo -e "${YELLOW}→${NC} Skipping cache backup"
fi

# Pull latest code (if using git)
if [ -d ".git" ]; then
    echo -e "${YELLOW}→${NC} Pulling latest code from git..."
    git pull
    echo -e "${GREEN}✓${NC} Code updated"
else
    echo -e "${YELLOW}→${NC} Not a git repository, skipping git pull"
fi

# Build and deploy with docker compose
echo -e "${YELLOW}→${NC} Stopping existing containers..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}→${NC} Building Docker images (no cache)..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache
else
    echo -e "${YELLOW}→${NC} Building Docker images..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml build
fi
echo -e "${GREEN}✓${NC} Images built"

echo -e "${YELLOW}→${NC} Starting containers in production mode..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
echo -e "${GREEN}✓${NC} Containers started"

# Wait for services to start
echo -e "${YELLOW}→${NC} Waiting for services to start..."
sleep 10

# Check if containers are running
if docker compose -f docker-compose.yml -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Containers are running"
else
    echo -e "${RED}✗${NC} Containers failed to start"
    echo "Check logs with: docker compose logs"
    exit 1
fi

# Test health endpoint with retry logic
echo -e "${YELLOW}→${NC} Testing health endpoint (may take up to 2 minutes for first deployment)..."
echo -e "${YELLOW}→${NC} If this is the first deployment, embeddings are being generated..."

HEALTH_CHECK_ATTEMPTS=0
MAX_ATTEMPTS=24  # 2 minutes total (24 * 5 seconds)

while [ $HEALTH_CHECK_ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Health check passed"
        break
    fi

    HEALTH_CHECK_ATTEMPTS=$((HEALTH_CHECK_ATTEMPTS + 1))

    if [ $HEALTH_CHECK_ATTEMPTS -lt $MAX_ATTEMPTS ]; then
        echo -ne "${YELLOW}→${NC} Waiting for app to be ready... (attempt $HEALTH_CHECK_ATTEMPTS/$MAX_ATTEMPTS)\r"
        sleep 5
    fi
done

if [ $HEALTH_CHECK_ATTEMPTS -eq $MAX_ATTEMPTS ]; then
    echo -e "\n${YELLOW}⚠${NC}  Health check timeout - app may still be starting"
    echo -e "${YELLOW}→${NC} This is normal on first deployment while embeddings are being generated"
    echo -e "${YELLOW}→${NC} Check status with: curl http://localhost:8000/api/health"
    echo -e "${YELLOW}→${NC} Monitor logs with: docker compose logs -f app"
fi

# Display container status
echo ""
echo -e "${BLUE}Container Status:${NC}"
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  View logs:        docker compose logs -f"
echo "  View app logs:    docker compose logs -f app"
echo "  View Redis logs:  docker compose logs -f redis"
echo "  Stop services:    docker compose -f docker-compose.yml -f docker-compose.prod.yml down"
echo "  Restart:          docker compose -f docker-compose.yml -f docker-compose.prod.yml restart"
echo ""
