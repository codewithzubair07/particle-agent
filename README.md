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

## Persona Cloning (Free Cloud Setup)

### What it does

- Attends meetings as you
- Speaks in your cloned voice via ElevenLabs
- Shows your face via Deep-Live-Cam (local)
- Thinks and responds using Gemini/OpenRouter LLM

### All free — setup steps

a. Hugging Face (STT):
   - Sign up at huggingface.co (free)
   - Get token from Settings → Access Tokens
   - Add to .env as HF_TOKEN

b. ElevenLabs (voice clone):
   - Sign up at elevenlabs.io (free tier = 10k chars/month)
   - Go to Voice Lab → Add Voice → clone your voice (needs 1 min of audio)
   - Copy the Voice ID
   - Add to .env as ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID

c. Face clone (local):
   - Save a clear photo as assets/my_face.jpg
   - git clone https://github.com/hacksider/Deep-Live-Cam
   - cd Deep-Live-Cam && pip install -r requirements.txt

d. Enable in config:
   - Set clone.enabled: true in config.yaml
   - For GPU: set clone.execution_provider: cuda

### Pipeline diagram

Meeting audio → HF Voxtral STT → Gemini LLM → ElevenLabs TTS (your voice) + Deep-Live-Cam (your face) → back into the meeting

## Usage

Run bootstrap validation:

```bash
python main.py
```

This loads `config.yaml` and `.env`, logs resolved settings with secrets masked, and exits.

## Deployment (Docker)

Run these from the `particle-agent/particle-agent` folder, after filling `.env`:

```bash
docker compose up -d --build
docker compose logs -f
```

The meeting bot is disabled in the image (`PARTICLE_DISABLE_MEETING_BOT=1`): it needs a visible Chrome window and audio devices.

## Deployment (systemd on Ubuntu)

1. Clone the repo to `/home/ubuntu/particle-agent` and create the virtual environment in `particle-agent/.venv`.
2. Install `libportaudio2`, then `pip install -r requirements.txt` and copy `.env.example` to `.env`.
3. Check the setup with `python main.py --validate`.
4. Install the service:
   ```bash
   sudo cp particle.service /etc/systemd/system/particle.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now particle.service
   ```
5. Inspect logs:
   ```bash
   journalctl -u particle.service -f
   ```

Never put passwords or tokens in `config.yaml`: use `.env` (for example `GOOGLE_MEET_PASSWORD`).

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
