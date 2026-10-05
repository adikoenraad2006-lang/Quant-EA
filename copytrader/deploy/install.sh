#!/usr/bin/env bash
# Install or update the copy trader on Ubuntu 24.04. Run from the repo checkout:
#     sudo bash copytrader/deploy/install.sh
# Safe to re-run: code and venv are refreshed, your config and secrets are kept.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # .../copytrader
APP=/opt/copytrader
ETC=/etc/copytrader
STATE=/var/lib/copytrader

[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

apt-get update -qq
apt-get install -y -qq python3 python3-venv >/dev/null

id copytrader &>/dev/null || useradd --system --home "$STATE" --shell /usr/sbin/nologin copytrader
install -d -o copytrader -g copytrader -m 750 "$STATE"
install -d -o root -g copytrader -m 750 "$ETC"
install -d -m 755 "$APP"

rm -rf "$APP/copytrader"
cp -r "$SRC" "$APP/copytrader"
find "$APP/copytrader" -name __pycache__ -prune -exec rm -rf {} +

[[ -d "$APP/venv" ]] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q -r "$APP/copytrader/requirements.txt"

if [[ ! -f "$ETC/config.toml" ]]; then
    install -o root -g copytrader -m 640 "$SRC/config.example.toml" "$ETC/config.toml"
    echo "created $ETC/config.toml - edit it"
fi
if [[ ! -f "$ETC/copytrader.env" ]]; then
    install -o root -g copytrader -m 640 "$SRC/deploy/copytrader.env.example" "$ETC/copytrader.env"
    echo "created $ETC/copytrader.env - put your password and API secret in it"
fi

install -m 644 "$SRC/deploy/copytrader.service" /etc/systemd/system/copytrader.service
systemctl daemon-reload

# Helper so every command below runs as the service user with the secrets loaded.
cat > /usr/local/bin/copytrader <<'WRAP'
#!/usr/bin/env bash
set -a; source /etc/copytrader/copytrader.env; set +a
cd /opt/copytrader
exec sudo -E -u copytrader /opt/copytrader/venv/bin/python -m copytrader "$@"
WRAP
chmod 755 /usr/local/bin/copytrader

echo
echo "installed. next:"
echo "  sudo nano $ETC/config.toml        # logins, leader, followers"
echo "  sudo nano $ETC/copytrader.env     # password + API secret"
echo "  sudo copytrader check             # test login, accounts, websocket"
echo "  sudo systemctl enable --now copytrader"
if systemctl is-active --quiet copytrader; then
    systemctl restart copytrader && echo "(service was running - restarted on the new code)"
fi
