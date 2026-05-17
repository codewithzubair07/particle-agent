# Particle Autonomous AI Assistant

Particle is a personal AI chief of staff designed for 24/7 autonomous operation.

## Installation

1. Install Python 3.11+ and create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment template:
   ```bash
   cp .env.example .env
   ```

## Configuration

- Edit `config.yaml` for runtime settings (paths, polling, module options).
- Fill all required keys in `.env`:
  - `GEMINI_API_KEY`
  - `OPENROUTER_API_KEY`
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_HOME_ID`
  - `EMAIL_ADDRESS`
  - `EMAIL_PASSWORD`
  - `GOOGLE_CALENDAR_CREDENTIALS`

## Usage

Run bootstrap validation:

```bash
python main.py
```

This loads `config.yaml` and `.env`, logs resolved settings with secrets masked, and exits.

## Deployment (Docker)

1. Build image:
   ```bash
   docker build -t particle-agent:latest .
   ```
2. Start service:
   ```bash
   docker compose up -d
   ```

## Deployment (systemd on Ubuntu 22.04)

1. Place project at `/home/ubuntu/particle-agent`.
2. Ensure virtual environment exists at `/home/ubuntu/particle-agent/.venv`.
3. Install service:
   ```bash
   sudo cp particle.service /etc/systemd/system/particle.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now particle.service
   ```
4. Inspect logs:
   ```bash
   journalctl -u particle.service -f
   ```

## Project Layout

```text
particle-agent/
├── main.py
├── cli.py
├── orchestrator.py
├── config.yaml
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── particle.service
├── modules/
├── context/
├── data/
└── logs/
```
