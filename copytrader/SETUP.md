# Tradovate copy trader — setup and testing guide

This guide takes you from nothing to a copier running 24/7 on a Hetzner CX23,
tested end to end on Tradovate's **demo** environment before any real money is
involved.

The copier watches one **leader** account and keeps each **follower** account at
the same position (times a multiplier, capped by a maximum). It mirrors
*positions*, not single fills, so a dropped connection or restart can never
leave a follower half-copied: the next check sees the gap and closes it.

```
Tradovate (Chicago)                     Hetzner CX23
 leader account  --- websocket push --->  copytrader  --- market orders ---> follower accounts
 follower accts  --- websocket push --->  (systemd)   --- Telegram alerts (optional)
```

Contents

1. [Get a Tradovate demo and API key](#1-get-a-tradovate-demo-and-api-key)
2. [Create the Hetzner server](#2-create-the-hetzner-server)
3. [Secure the server](#3-secure-the-server)
4. [Install the copier](#4-install-the-copier)
5. [Configure it](#5-configure-it)
6. [Test it, stage by stage](#6-test-it-stage-by-stage)
7. [Run it as a service](#7-run-it-as-a-service)
8. [Day-to-day operation](#8-day-to-day-operation)
9. [Going live checklist](#9-going-live-checklist)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Get a Tradovate demo and API key

### 1a. The demo account

* Go to **https://www.tradovate.com** and pick the free demo / simulated
  trading option (or open **https://trader.tradovate.com** and sign up there).
  You get a simulated account with fake money that trades real CME prices.
* The free demo has historically lasted **14 days**. Pick your testing window
  accordingly, or ask Tradovate support about extending it.
* If you already have a live Tradovate account, you usually already have
  simulated accounts under the same login. Log in at trader.tradovate.com and
  switch to **Simulation** in the account menu.

### 1b. You need two accounts to copy between

Copying needs a leader and at least one follower. Options:

* **One login with two sim accounts.** Some logins have, or can add, a second
  simulated account. Run `copytrader check` (step 6) and it lists every
  account your login can see.
* **Two logins.** Sign up for a second demo with a different email address. Put
  it in the config as a second `[logins.*]` section. The copier supports leader
  and followers on different logins.

### 1c. The API key

The copier logs in through Tradovate's official API, which needs an API key (a
numeric **cid** and a **secret**).

* In the Tradovate web app: **Settings (gear icon) → API Access → Generate API
  Key**. Give it these permissions: **Orders: Full Access**, **Positions**,
  **Account Information**, **Contract Library**. The secret is shown **once**,
  so copy it straight away.
* At the time of writing, Tradovate only lets you create API keys after you buy
  the **API Access add-on**, and that has required a **funded live account**.
  A key created this way works against both the demo and live API. Check
  Tradovate's current API access terms before you pay for anything; they change.
* If your follower is on a second login, that login may need its own key. If
  `check` reports a login refused for the second user, generate a key under
  that user and put it in that login's section.

> **Prop-firm accounts (Apex, Topstep, Take Profit Trader, etc.):** these run on
> Tradovate but most do **not** give you API access, and many forbid automated
> trading, VPS use, or copying between accounts held by different people. Read
> your firm's rules before pointing this at a funded or evaluation account.

---

## 2. Create the Hetzner server

1. **Make an SSH key** on your own computer, if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "copytrader"
   cat ~/.ssh/id_ed25519.pub          # copy this line
   ```
   (Windows: run the same commands in PowerShell.)
2. Open **https://console.hetzner.cloud** → create a project → **Add Server**:
   * **Location:** Nuremberg, Falkenstein or Helsinki (the CX line is EU only).
     Orders take ~100–130 ms to reach Chicago from there, which is fine unless
     you scalp a few ticks. For lower latency use **Ashburn, VA** with a
     **CPX11** instead (~25 ms). Everything in this guide works the same.
   * **Image:** Ubuntu 24.04
   * **Type:** Shared vCPU → x86 → **CX23**
   * **Networking:** keep **Public IPv4** on
   * **SSH keys:** paste the public key from step 1
   * **Firewalls:** create one that allows **inbound TCP 22 only**, ideally from
     your home IP only. The copier makes outgoing connections only, so it needs
     no open inbound ports.
   * **Name:** `copytrader`
3. Note the server's IP address and connect:
   ```bash
   ssh root@YOUR_SERVER_IP
   ```

---

## 3. Secure the server

Run these as root on the server, one block at a time.

```bash
# Updates, plus automatic security updates from now on
apt update && apt upgrade -y
apt install -y unattended-upgrades ufw git
dpkg-reconfigure -f noninteractive unattended-upgrades

# Your own admin user (replace "trader" with any name)
adduser trader
usermod -aG sudo trader
rsync --archive --chown=trader:trader ~/.ssh /home/trader
```

**Open a second terminal now** and check that `ssh trader@YOUR_SERVER_IP` works
and that `sudo whoami` prints `root`. Only then lock down SSH:

```bash
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh

# Server firewall (on top of the Hetzner one)
ufw allow OpenSSH
ufw --force enable

# Accurate time matters for login tokens; this should say "System clock synchronized: yes"
timedatectl
```

From here on, log in as `trader` and use `sudo`.

**Check the server can reach Tradovate:**

```bash
curl -s -X POST https://demo.tradovateapi.com/v1/auth/accesstokenrequest \
     -H 'Content-Type: application/json' -d '{"name":"x","password":"x"}'
```

A JSON reply with an `errorText` (wrong username or password) is the good
result: it means you got through to Tradovate. A timeout means a network
problem.

---

## 4. Install the copier

Get the code onto the server. Pick one of these:

```bash
# A) Clone the repo (private repo: use a GitHub personal access token as the password,
#    or add a read-only deploy key to the repo)
git clone https://github.com/adikoenraad2006-lang/quant-ea.git ~/quant-ea

# B) Or copy just the copytrader folder from your computer (run these ON YOUR COMPUTER,
#    from inside your Quant-EA folder)
ssh trader@YOUR_SERVER_IP mkdir -p quant-ea
scp -r copytrader trader@YOUR_SERVER_IP:~/quant-ea/
```

Then install:

```bash
cd ~/quant-ea
sudo bash copytrader/deploy/install.sh
```

The installer:

* installs Python and creates a locked-down `copytrader` system user
* puts the code in `/opt/copytrader` with its own virtualenv
* creates `/etc/copytrader/config.toml` and `/etc/copytrader/copytrader.env`
  (only if they don't exist yet, so re-running it never overwrites your settings)
* installs the `copytrader` systemd service, but **does not start it**
* adds a `copytrader` command for the manual steps below

---

## 5. Configure it

**Secrets** go in the env file, so they are not in the main config:

```bash
sudo nano /etc/copytrader/copytrader.env
```
```
TRADOVATE_MAIN_PASSWORD=your-demo-password
TRADOVATE_MAIN_SEC=your-api-secret
```

**Everything else:**

```bash
sudo nano /etc/copytrader/config.toml
```

The minimum to fill in:

| setting | what to put |
| --- | --- |
| `environment` | `"demo"` while testing |
| `dry_run` | leave `true` for now |
| `[logins.main] username` | your Tradovate username |
| `[logins.main] cid` | the numeric API key ID |
| `[leader] account` | the leader account's name, e.g. `DEMO1234567` |
| `[[followers]] account` | the follower account's name |
| `multiplier` | `1.0` = same size; `0.5` = half (rounded down toward zero) |
| `max_position` | the most contracts the follower may ever hold. Keep it small. |

If you don't know the exact account names yet, fill in the login details and
leave the account names as they are. `check` in the next step logs in, prints
every account it can see, and flags the names that don't match. Copy the right
ones in and run it again.

Save in nano with **Ctrl+O, Enter**, then exit with **Ctrl+X**.

---

## 6. Test it, stage by stage

Do these in order. Each stage only risks what the one before it proved safe.
Make sure the service is **not** running during stages 1–4
(`sudo systemctl stop copytrader`). The copier also refuses to start a second
copy of itself.

### Stage 0: the built-in tests (no Tradovate needed)

```bash
sudo -u copytrader /opt/copytrader/venv/bin/python /opt/copytrader/copytrader/tests/test_copytrader.py
```

This should end with `ALL PASSED`. It covers the copy logic (opens, adds,
reversals, multiplier and cap, pause file, retry and rate breaker) and runs the
real client against a fake Tradovate server, including a dropped connection.

### Stage 1: connection check

```bash
sudo copytrader check
```

Expected output:

```
[main] logging in as yourname ...
  OK    logged in, user id 1234567
        account DEMO1234567          id 9876543    flat                    <- leader
        account DEMO7654321          id 9876544    flat                    <- follower x1.0 max 2
  OK    websocket authorized and synced

ALL CHECKS PASSED
```

Fix anything that says FAIL before you go further (see
[Troubleshooting](#10-troubleshooting)). Make sure **both accounts are flat**
before you continue.

### Stage 2: dry run (sees everything, sends nothing)

With `dry_run = true` still set:

```bash
sudo copytrader run
```

Leave it running. In the Tradovate web app, on the **leader** account, buy
**1 MES** (Micro E-mini S&P, the cheapest contract to test with). Within about
a second the server log should show:

```
DRY RUN would send main/DEMO7654321: Buy 1 MESZ6  (leader +1 -> target +1, follower +0)
```

Close the leader position and you'll see the matching `Sell`. Stop with
**Ctrl+C**.

### Stage 3: real orders on demo

Set `dry_run = false` in the config, then `sudo copytrader run` again. Keep the
Tradovate web app open on **both** accounts and work through this table. After
each step, check that the follower shows the expected position.

| # | do this on the leader | follower should be (multiplier 1, max 2) |
| --- | --- | --- |
| 1 | buy 1 MES | +1 |
| 2 | buy 1 more | +2 |
| 3 | buy 1 more (now +3) | +2 (capped by `max_position`) |
| 4 | sell 1 (now +2) | +2 |
| 5 | sell 4 (reverse to −2) | −2 |
| 6 | buy 2 (flat) | flat |
| 7 | buy 1 MES **and** 1 MNQ | +1 MES, +1 MNQ |
| 8 | flatten all | flat on both |

Then try a multiplier: set `multiplier = 0.5`, restart, and check that leader
+3 gives follower +1, and leader +1 gives follower 0.

### Stage 4: the safety features

**Restart while in a position.** Leader buys 1 and the follower copies it.
Press Ctrl+C, then leader buys 1 more (now +2), then start the copier again.
The follower should go to +2 within a few seconds.

**Connection loss.** In a second SSH window, block outgoing HTTPS for 30
seconds (your SSH session is not affected):

```bash
sudo ufw insert 1 deny out 443/tcp
# meanwhile: change the leader's position in the web app
sleep 30; sudo ufw delete deny out 443/tcp
```

The log shows `socket down` and `reconnecting`, then catches the follower up.

**Pause switch.** `sudo touch /var/lib/copytrader/PAUSE`, then trade the leader.
The log says `PAUSED` and sends nothing. `sudo rm /var/lib/copytrader/PAUSE`
and the follower catches up straight away.

**Emergency flatten.**

```bash
sudo touch /var/lib/copytrader/PAUSE       # first, so the copier doesn't re-open them
sudo copytrader flatten
```

It lists every follower position and asks you to type `YES` before closing
them all at market.

**Start-up with an open leader position.** Open a leader position, then start
the copier. By default (`adopt_on_start = false`) it leaves that position alone
and only mirrors once the leader changes it. Read the log line that explains
this, so it doesn't surprise you later.

---

## 7. Run it as a service

Once stage 4 passes:

```bash
sudo systemctl enable --now copytrader     # start now and on every boot
systemctl status copytrader                # should say "active (running)"
journalctl -u copytrader -f                # live log; Ctrl+C to stop watching
```

systemd restarts it within 5 seconds if it crashes. Test that with
`sudo reboot`: after the server comes back, `systemctl status copytrader`
should be running again with no action from you.

Let it copy on demo for **at least a few full trading days**, including a
Sunday-evening open and a daily maintenance break, before you think about
going live.

### Optional: Telegram alerts

You get a message on start/stop, every order sent, failures, pauses and halts.

1. In Telegram, message **@BotFather** → `/newbot` → copy the **bot token**.
2. Send any message to your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy the
   `"chat":{"id": ...}` number.
3. Put both in the `[telegram]` section of the config, then:
   ```bash
   sudo copytrader alert                  # sends a test message
   sudo systemctl restart copytrader
   ```

---

## 8. Day-to-day operation

| task | command |
| --- | --- |
| live log | `journalctl -u copytrader -f` |
| today's orders | `journalctl -u copytrader --since today \| grep -E "sent\|FAILED\|HALTED"` |
| status | `systemctl status copytrader` |
| pause copying | `sudo touch /var/lib/copytrader/PAUSE` |
| resume | `sudo rm /var/lib/copytrader/PAUSE` |
| close all followers | pause first, then `sudo copytrader flatten` |
| after editing config | `sudo systemctl restart copytrader` |
| stop completely | `sudo systemctl stop copytrader` |
| update the code | `cd ~/quant-ea && git pull && sudo bash copytrader/deploy/install.sh` |

**If a follower gets HALTED** (more than `max_orders_per_minute` orders in a
minute), something is fighting the copier. Common causes are a manual trade on
the follower, or an order being rejected for margin. Look at both accounts,
fix the cause, then `sudo systemctl restart copytrader`.

**Don't trade the follower accounts by hand** while the copier runs: it will
undo your trade on the next pass to match the leader. Pause first.

---

## 9. Going live checklist

* [ ] Several days of clean demo copying, with no `FAILED` or `HALTED` in the log
* [ ] You've read the rules of every firm or broker involved about API use and copy trading
* [ ] The API key works against live: set `environment = "live"`, keep `dry_run = true`, run `sudo copytrader check`
* [ ] Live account names are in the config, and `max_position` is set to what you'd accept losing in a gap
* [ ] Telegram alerts work (`sudo copytrader alert`)
* [ ] One live dry-run session watched end to end
* [ ] Only then set `dry_run = false` and restart, and watch the first live trades closely

---

## 10. Troubleshooting

| symptom | likely cause and fix |
| --- | --- |
| `config error: ...` and the service won't stay up | The message names the problem. Fix the config and `sudo systemctl start copytrader`. |
| `login refused: Incorrect username or password` | Check username and password. The password goes in `copytrader.env`. |
| login refused, mentions the app or the key | Wrong `cid` or `sec`, the key lacks permissions, or the API add-on isn't active. |
| `Tradovate wants a captcha` | Too many failed logins. Log in once through the web app, wait an hour, try again. |
| `login rate-limited, retrying` | Normal after several quick restarts; it waits and retries by itself. |
| `account 'X' not found` | Use the exact name `check` prints (case matters). |
| `check` passes on demo, fails on live | The API key or the add-on isn't enabled for live, or the account names differ. |
| socket keeps reconnecting | Check `timedatectl` (clock), the server's internet (`curl` test in step 3), and Tradovate's status page. |
| `ORDER FAILED ... rejected` | Read the reason: usually margin, a contract outside trading hours, or account permissions. The copier retries after `order_timeout`. |
| follower doesn't copy a position that existed before start-up | Expected with `adopt_on_start = false`. It mirrors once the leader changes that position. |
| `another copier is already running` | The service is running; `sudo systemctl stop copytrader` before running it by hand. |
