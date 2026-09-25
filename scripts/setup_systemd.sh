#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="/opt/voice-agent"
SERVICE_FILE="/etc/systemd/system/voice-agent.service"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: setup_systemd.sh must be run as root (use sudo ./deploy.sh)" >&2
    exit 1
fi

RUN_AS="${SUDO_USER:-root}"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Hybrid Full-Duplex Voice Agent
After=network.target

[Service]
Type=simple
User=${RUN_AS}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable voice-agent
systemctl restart voice-agent

echo "==> Systemd service installed and started"
echo "    Status:  systemctl status voice-agent"
echo "    Logs:    journalctl -u voice-agent -f"
echo "    UI:      http://$(hostname -I | awk '{print $1}'):8000"
