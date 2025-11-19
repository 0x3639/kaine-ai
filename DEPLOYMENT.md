# Kaine AI Production Deployment Guide

This guide walks you through deploying Kaine AI to production with Docker, Redis, and Caddy reverse proxy.

## Architecture Overview

```
Internet
    ↓
Caddy Server (Separate Server)
    - SSL/TLS termination (automatic Let's Encrypt)
    - Reverse proxy
    - Security headers
    - Access logging
    ↓
Application Server
    - Docker containers:
      - FastAPI app (Kaine AI)
      - Redis (rate limiting)
    - Volumes for cache persistence
```

## Prerequisites

### Application Server Requirements

- **OS**: Ubuntu 20.04+ or Debian 11+ (recommended)
- **RAM**: 2GB minimum, 4GB+ recommended
- **CPU**: 2+ cores recommended
- **Disk**: 10GB+ free space
- **Software**:
  - Docker Engine 20.10+
  - Docker Compose 2.0+
  - Git (optional, for updates)

### Caddy Server Requirements

- **OS**: Ubuntu 20.04+ or Debian 11+
- **RAM**: 512MB minimum
- **CPU**: 1 core minimum
- **Disk**: 5GB free space
- **Software**: Caddy 2.6+
- **Network**: Port 80 and 443 open to internet

### Other Requirements

- Domain name pointing to Caddy server IP
- OpenAI API key
- SSH access to both servers

---

## Part 1: Application Server Setup

### Step 1: Install Docker and Docker Compose

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (logout/login required after this)
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

### Step 2: Clone and Configure the Application

```bash
# Create application directory
mkdir -p ~/kaine-ai
cd ~/kaine-ai

# Clone your repository (or upload files via SCP)
git clone https://github.com/0x3639/kaine-ai.git .

# Or upload files manually:
# scp -r /local/path/to/kaine-ai/* user@your-server:~/kaine-ai/
```

### Step 3: Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Update these critical variables in `.env`:**

```bash
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Production mode
ENVIRONMENT=production

# Server configuration
PORT=8000
WORKERS=4  # Adjust based on CPU cores (typically 2-4x cores)
LOG_LEVEL=INFO  # Use WARNING or ERROR in production for less noise

# Redis (using Docker service name)
REDIS_URL=redis://redis:6379/0

# CORS - IMPORTANT: Replace with your actual domain
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Rate limiting
RATE_LIMIT_MAX_REQUESTS=10
RATE_LIMIT_WINDOW_MINUTES=60
```

**Save and exit** (Ctrl+X, then Y, then Enter in nano)

### Step 4: Prepare Data Files

```bash
# Ensure your data directory exists
mkdir -p data

# Upload your Telegram posts JSON file
# Option 1: Copy from local machine
scp /local/path/to/mrkainez_posts.json user@your-server:~/kaine-ai/data/

# Option 2: Download from URL
# wget -O data/mrkainez_posts.json https://your-url/posts.json

# Verify file exists
ls -lh data/
```

### Step 5: Create Cache Directory

```bash
# Create cache directory for embeddings
mkdir -p cache

# Set proper permissions
chmod 755 cache
```

### Step 6: Deploy with Docker Compose

```bash
# Build and start services
./deploy.sh

# Or manually:
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

**The deployment script will:**
- Validate environment configuration
- Backup existing cache (if any)
- Pull latest code (if git repo)
- Build Docker images
- Start containers
- Run health checks

### Step 7: Verify Application is Running

```bash
# Check container status
docker compose ps

# Should show both containers running:
# - kaine-app (healthy)
# - kaine-redis (healthy)

# View logs
docker compose logs -f app

# Test health endpoint
curl http://localhost:8000/api/health

# Expected output:
# {
#   "status": "healthy",
#   "environment": "production",
#   "posts_loaded": <number>,
#   "rate_limit": "10 per 60 minutes",
#   "dependencies": {
#     "redis": "healthy",
#     "qa_tool": "healthy"
#   }
# }
```

### Step 8: Configure Firewall

```bash
# Install UFW if not present
sudo apt install -y ufw

# Allow SSH (IMPORTANT: do this first!)
sudo ufw allow ssh

# Application runs on localhost only - no external access needed
# Caddy on separate server will connect via internal network or VPN

# If you need to allow Caddy server to connect:
# Replace CADDY_SERVER_IP with your Caddy server's IP
sudo ufw allow from CADDY_SERVER_IP to any port 8000 proto tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

---

## Part 2: Caddy Server Setup

### Step 1: Install Caddy

On your **separate Caddy server**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl

# Add Caddy repository
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list

# Install Caddy
sudo apt update
sudo apt install -y caddy

# Verify installation
caddy version
```

### Step 2: Configure Caddy

```bash
# Backup default Caddyfile
sudo mv /etc/caddy/Caddyfile /etc/caddy/Caddyfile.backup

# Create new Caddyfile
sudo nano /etc/caddy/Caddyfile
```

**Paste the following configuration:**

```
# Replace yourdomain.com with your actual domain
# Replace YOUR_APP_SERVER_IP with your application server's IP

yourdomain.com {
    encode gzip zstd

    log {
        output file /var/log/caddy/kaine-ai-access.log {
            roll_size 100mb
            roll_keep 5
            roll_keep_for 720h
        }
        format json
    }

    reverse_proxy YOUR_APP_SERVER_IP:8000 {
        health_uri /api/health
        health_interval 30s
        health_timeout 10s

        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
        header_up X-Forwarded-Host {host}

        transport http {
            dial_timeout 10s
            response_header_timeout 60s
            read_timeout 120s
        }

        fail_duration 30s
        max_fails 3
        unhealthy_status 503
    }

    header {
        -Server
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
        Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    }
}

# Redirect www to non-www
www.yourdomain.com {
    redir https://yourdomain.com{uri} permanent
}
```

**Save and exit** (Ctrl+X, then Y, then Enter)

### Step 3: Create Log Directory

```bash
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy
```

### Step 4: Configure DNS

**Before starting Caddy**, ensure your domain DNS is configured:

1. Add an **A record** pointing `yourdomain.com` to your Caddy server IP
2. Add an **A record** pointing `www.yourdomain.com` to your Caddy server IP
3. Wait for DNS propagation (check with `dig yourdomain.com`)

### Step 5: Configure Firewall

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow ssh

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### Step 6: Start Caddy

```bash
# Test configuration
sudo caddy validate --config /etc/caddy/Caddyfile

# Start Caddy service
sudo systemctl start caddy

# Enable auto-start on boot
sudo systemctl enable caddy

# Check status
sudo systemctl status caddy

# View logs
sudo journalctl -u caddy -f
```

Caddy will automatically obtain Let's Encrypt SSL certificates for your domain.

### Step 7: Verify HTTPS is Working

```bash
# Test from Caddy server
curl https://yourdomain.com/api/health

# Expected: JSON response with status "healthy"

# Check SSL certificate
curl -vI https://yourdomain.com 2>&1 | grep -i "subject:"
```

---

## Part 3: Testing and Verification

### Test the Full Stack

1. **Access the web interface**: Open `https://yourdomain.com` in your browser
2. **Ask a test question**: Type a question and verify you get a response
3. **Check rate limiting**: Make 11 requests quickly and verify the 11th is blocked
4. **Monitor logs**: Watch logs on both servers during testing

### Application Server Testing

```bash
cd ~/kaine-ai

# Check container health
docker compose ps

# View application logs
docker compose logs -f app | grep -i error

# View Redis logs
docker compose logs -f redis

# Test Redis connection
docker compose exec redis redis-cli ping
# Expected: PONG

# Monitor resource usage
docker stats
```

### Caddy Server Testing

```bash
# Check Caddy status
sudo systemctl status caddy

# View access logs
sudo tail -f /var/log/caddy/kaine-ai-access.log

# View Caddy service logs
sudo journalctl -u caddy -f

# Test reverse proxy
curl -I https://yourdomain.com
```

---

## Part 4: Monitoring and Maintenance

### Application Logs

```bash
# View all logs
docker compose logs -f

# View last 100 lines of app logs
docker compose logs --tail=100 app

# Search for errors
docker compose logs app | grep -i error

# Export logs
docker compose logs app > app_logs_$(date +%Y%m%d).log
```

### Health Monitoring

```bash
# Check health endpoint
curl https://yourdomain.com/api/health | jq

# Monitor health continuously
watch -n 5 'curl -s https://yourdomain.com/api/health | jq'
```

### Updates and Redeployment

```bash
# On application server
cd ~/kaine-ai

# Pull latest code
git pull

# Redeploy with backup
./deploy.sh

# Or redeploy without backup and force rebuild
./deploy.sh --no-backup --rebuild
```

### Backup Strategy

**Critical files to backup:**

```bash
# Backup script
#!/bin/bash
BACKUP_DIR=~/kaine-backups/$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup cache (embeddings - expensive to regenerate)
cp -r ~/kaine-ai/cache $BACKUP_DIR/

# Backup environment
cp ~/kaine-ai/.env $BACKUP_DIR/

# Backup data
cp -r ~/kaine-ai/data $BACKUP_DIR/

# Backup Redis data
docker compose exec redis redis-cli SAVE
docker cp kaine-redis:/data/dump.rdb $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

### Restore from Backup

```bash
# Restore cache
cp -r /path/to/backup/cache ~/kaine-ai/

# Restore Redis data
docker compose down
cp /path/to/backup/dump.rdb ~/kaine-ai/redis_data/
docker compose up -d

# Rebuild if needed
./deploy.sh --rebuild
```

---

## Part 5: Troubleshooting

### Application Won't Start

```bash
# Check environment variables
docker compose config

# Check logs for errors
docker compose logs app | tail -50

# Common issues:
# 1. Missing OPENAI_API_KEY
cat .env | grep OPENAI_API_KEY

# 2. Missing data file
ls -la data/

# 3. Port already in use
sudo netstat -tlnp | grep 8000

# 4. Permissions issue
sudo chown -R $USER:$USER cache data
```

### Redis Connection Issues

```bash
# Check Redis is running
docker compose ps redis

# Test Redis connection
docker compose exec redis redis-cli ping

# Check Redis logs
docker compose logs redis

# Restart Redis
docker compose restart redis
```

### Rate Limiting Not Working

```bash
# Check Redis rate limit keys
docker compose exec redis redis-cli KEYS "rate_limit:*"

# View rate limit data for specific IP
docker compose exec redis redis-cli ZRANGE "rate_limit:1.2.3.4" 0 -1 WITHSCORES

# Clear rate limits (for testing)
docker compose exec redis redis-cli FLUSHDB
```

### High Memory Usage

```bash
# Check container memory
docker stats --no-stream

# Reduce workers in .env
nano .env
# Set WORKERS=2

# Restart
docker compose restart app

# Configure Redis memory limit
docker compose exec redis redis-cli CONFIG SET maxmemory 256mb
```

### SSL Certificate Issues (Caddy)

```bash
# Check certificate status
sudo caddy trust

# View certificate info
sudo caddy trust list

# Force certificate renewal
sudo systemctl restart caddy

# Check Let's Encrypt logs
sudo journalctl -u caddy | grep -i acme
```

### Connection Refused from Caddy to App

```bash
# On app server - check firewall
sudo ufw status

# Test from Caddy server
curl http://APP_SERVER_IP:8000/api/health

# Check docker is listening on correct interface
sudo netstat -tlnp | grep 8000

# Verify docker-compose.prod.yml port mapping
cat docker-compose.prod.yml | grep ports
```

---

## Part 6: Performance Optimization

### Scaling Workers

```bash
# Edit .env
nano .env

# Adjust based on CPU cores
# Formula: workers = (2 x cores) + 1
# For 4 cores: WORKERS=9
# Start conservative: WORKERS=4

# Apply changes
docker compose restart app
```

### Redis Tuning

```bash
# Persistent connection pooling
docker compose exec redis redis-cli CONFIG SET tcp-keepalive 300

# Memory optimization
docker compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
docker compose exec redis redis-cli CONFIG SET maxmemory 512mb

# Save configuration
docker compose exec redis redis-cli CONFIG REWRITE
```

### Application Optimization

```bash
# Pre-generate embeddings for faster startup
# On first deployment, let it complete initialization
# The cache/ directory will be populated

# For updates without changing data:
# Don't delete cache/ - embeddings will be reused
```

---

## Security Best Practices

1. **Keep secrets secure**
   - Never commit `.env` to git
   - Use strong, unique OpenAI API keys
   - Rotate keys periodically

2. **Update regularly**
   ```bash
   sudo apt update && sudo apt upgrade -y
   docker compose pull
   ./deploy.sh --rebuild
   ```

3. **Monitor access logs**
   ```bash
   sudo tail -f /var/log/caddy/kaine-ai-access.log | jq
   ```

4. **Restrict SSH access**
   ```bash
   # Use SSH keys only
   sudo nano /etc/ssh/sshd_config
   # Set: PasswordAuthentication no
   sudo systemctl restart ssh
   ```

5. **Configure rate limits**
   - Adjust `RATE_LIMIT_MAX_REQUESTS` based on usage
   - Monitor for abuse in logs
   - Consider IP whitelisting for known users

---

## Cost Considerations

### OpenAI API Costs

- **Embeddings** (one-time): ~$0.13 per 1M tokens
- **Per query**: ~$0.02 (without compression)
- **With compression enabled**: ~$0.22 per query

**Monitor costs**:
```bash
# Enable cost tracking in .env
ENABLE_COST_TRACKING=true

# View logs to see per-query costs
docker compose logs app | grep -i cost
```

### Server Costs

- **Application Server**: $10-20/month (2GB RAM VPS)
- **Caddy Server**: $5-10/month (512MB RAM VPS)
- **Total estimated**: $15-30/month

---

## Support and Resources

- **Logs location (app)**: `docker compose logs app`
- **Logs location (Caddy)**: `/var/log/caddy/kaine-ai-access.log`
- **Health endpoint**: `https://yourdomain.com/api/health`
- **Caddy documentation**: https://caddyserver.com/docs/
- **Docker Compose docs**: https://docs.docker.com/compose/

---

## Quick Reference Commands

```bash
# Application Server
cd ~/kaine-ai
./deploy.sh                          # Deploy/redeploy
docker compose ps                    # Check status
docker compose logs -f app           # View logs
docker compose restart app           # Restart app
docker compose down                  # Stop all services
docker compose up -d                 # Start services

# Caddy Server
sudo systemctl status caddy          # Check status
sudo systemctl restart caddy         # Restart Caddy
sudo journalctl -u caddy -f          # View logs
sudo caddy reload --config /etc/caddy/Caddyfile  # Reload config
```

---

**Congratulations!** 🎉 Your Kaine AI is now running in production with:
- ✅ Automatic HTTPS via Let's Encrypt
- ✅ Redis-backed rate limiting
- ✅ Structured JSON logging
- ✅ Health monitoring
- ✅ Production-grade security
- ✅ Easy deployment and updates
