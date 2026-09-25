#!/usr/bin/env bash
# Configure nginx + Let's Encrypt HTTPS for the voice agent.
#
# Usage:
#   sudo DOMAIN=voice.metl.run CERTBOT_EMAIL=you@example.com ./scripts/setup_nginx.sh
#
# Prerequisites:
#   - voice-agent systemd service running on 127.0.0.1:8000 (sudo ./deploy.sh)
#   - DNS A record: voice.metl.run -> this server's public IP

set -euo pipefail

DOMAIN="${DOMAIN:-voice.metl.run}"
EMAIL="${CERTBOT_EMAIL:-}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_TEMPLATE="$SCRIPT_DIR/nginx/voice-agent.conf.template"
SSL_TEMPLATE="$SCRIPT_DIR/nginx/voice-agent-ssl.conf.template"
SITE_AVAILABLE="/etc/nginx/sites-available/voice-agent"
SITE_ENABLED="/etc/nginx/sites-enabled/voice-agent"
WEBROOT="/var/www/certbot"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root: sudo ./scripts/setup_nginx.sh" >&2
    exit 1
fi

if [[ ! -f "$NGINX_TEMPLATE" || ! -f "$SSL_TEMPLATE" ]]; then
    echo "ERROR: nginx templates not found in $SCRIPT_DIR/nginx/" >&2
    exit 1
fi

echo "==> Domain: $DOMAIN"
echo "==> Upstream: 127.0.0.1:$UPSTREAM_PORT"

# ---------------------------------------------------------------------------
# DNS check (best-effort)
# ---------------------------------------------------------------------------
SERVER_IP="$(curl -4 -sf --max-time 5 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"
RESOLVED_IP="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || true)"
if [[ -n "$RESOLVED_IP" && -n "$SERVER_IP" && "$RESOLVED_IP" != "$SERVER_IP" ]]; then
    echo "WARNING: $DOMAIN resolves to $RESOLVED_IP but this server is $SERVER_IP"
    echo "         Fix DNS before continuing, or certbot will fail."
    if [[ "${CONTINUE_ON_DNS_MISMATCH:-}" != "1" ]]; then
        if [[ -t 0 ]]; then
            read -r -p "Continue anyway? [y/N] " ans
            [[ "${ans,,}" == "y" ]] || exit 1
        else
            echo "Set CONTINUE_ON_DNS_MISMATCH=1 to skip this check." >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Install packages
# ---------------------------------------------------------------------------
echo "==> Installing nginx and certbot"
apt-get update -qq
apt-get install -y -qq nginx certbot

# python3-certbot-nginx is optional; we manage configs ourselves
if apt-cache show python3-certbot-nginx &>/dev/null; then
    apt-get install -y -qq python3-certbot-nginx 2>/dev/null || true
fi

mkdir -p "$WEBROOT"

# ---------------------------------------------------------------------------
# HTTP-only config first (needed for certbot webroot / nginx plugin)
# ---------------------------------------------------------------------------
echo "==> Writing temporary HTTP nginx config"
sed "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" "$NGINX_TEMPLATE" > "$SITE_AVAILABLE"
sed -i "s/127.0.0.1:8000/127.0.0.1:$UPSTREAM_PORT/g" "$SITE_AVAILABLE"

ln -sf "$SITE_AVAILABLE" "$SITE_ENABLED"
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl enable nginx
systemctl reload nginx

# ---------------------------------------------------------------------------
# Firewall (if ufw is active)
# ---------------------------------------------------------------------------
if command -v ufw &>/dev/null && ufw status | grep -q "Status: active"; then
    echo "==> Opening ports 80 and 443 in ufw"
    ufw allow 80/tcp
    ufw allow 443/tcp
fi

# ---------------------------------------------------------------------------
# Obtain TLS certificate
# ---------------------------------------------------------------------------
echo "==> Requesting Let's Encrypt certificate for $DOMAIN"
CERTBOT_ARGS=(certonly --webroot -w "$WEBROOT" -d "$DOMAIN" --non-interactive --agree-tos)
if [[ -n "$EMAIL" ]]; then
    CERTBOT_ARGS+=(--email "$EMAIL")
else
    echo "    (no CERTBOT_EMAIL set — using register-unsafely-without-email)"
    CERTBOT_ARGS+=(--register-unsafely-without-email)
fi

if ! certbot "${CERTBOT_ARGS[@]}"; then
    echo ""
    echo "ERROR: certbot failed. Common fixes:" >&2
    echo "  1. Ensure DNS A record for $DOMAIN points to this server ($SERVER_IP)" >&2
    echo "  2. Ensure ports 80/443 are open in your cloud firewall" >&2
    echo "  3. Wait a few minutes for DNS propagation and retry" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Final HTTPS config
# ---------------------------------------------------------------------------
echo "==> Installing HTTPS nginx config"
sed "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" "$SSL_TEMPLATE" > "$SITE_AVAILABLE"
sed -i "s/127.0.0.1:8000/127.0.0.1:$UPSTREAM_PORT/g" "$SITE_AVAILABLE"

nginx -t
systemctl reload nginx

# Ensure certbot renewal timer is active
systemctl enable certbot.timer 2>/dev/null || true

echo ""
echo "==> HTTPS setup complete"
echo "    URL:     https://$DOMAIN"
echo "    Test:    curl -I https://$DOMAIN"
echo "    Nginx:   systemctl status nginx"
echo "    Renew:   certbot renew --dry-run"
echo ""
echo "    Tip: close public access to port $UPSTREAM_PORT if exposed:"
echo "         sudo ufw deny $UPSTREAM_PORT/tcp"
