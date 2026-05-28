Particle Autonomous AI Assistant
Particle is a personal AI chief of staff designed for 24/7 autonomous operation.

Installation
Install Python 3.11+ and create a virtual environment.
Install dependencies:
pip install -r requirements.txt
Copy environment template:
cp .env.example .env
Configuration
Edit config.yaml for runtime settings (paths, polling, module options).
Fill all required keys in .env:
GEMINI_API_KEY
OPENROUTER_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_HOME_ID
EMAIL_ADDRESS
EMAIL_PASSWORD
GOOGLE_CALENDAR_CREDENTIALS
Persona Cloning (Free Cloud Setup)
What it does
Attends meetings as you
Speaks in your cloned voice via ElevenLabs
Shows your face via Deep-Live-Cam (local)
Thinks and responds using Gemini/OpenRouter LLM
All free — setup steps
a. Hugging Face (STT):

Sign up at huggingface.co (free)
Get token from Settings → Access Tokens
Add to .env as HF_TOKEN
b. ElevenLabs (voice clone):

Sign up at elevenlabs.io (free tier = 10k chars/month)
Go to Voice Lab → Add Voice → clone your voice (needs 1 min of audio)
Copy the Voice ID
Add to .env as ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID
c. Face clone (local):

Save a clear photo as assets/my_face.jpg
git clone https://github.com/hacksider/Deep-Live-Cam
cd Deep-Live-Cam && pip install -r requirements.txt
d. Enable in config:

Set clone.enabled: true in config.yaml
For GPU: set clone.execution_provider: cuda
Pipeline diagram
Meeting audio → HF Voxtral STT → Gemini LLM → ElevenLabs TTS (your voice) + Deep-Live-Cam (your face) → back into the meeting

Usage
Run bootstrap validation:

python main.py
This loads config.yaml and .env, logs resolved settings with secrets masked, and exits.

Deployment (Docker)
Build image:
docker build -t particle-agent:latest .
Start service:
docker compose up -d
Deployment (systemd on Ubuntu 22.04)
Place project at /home/ubuntu/particle-agent.
Ensure virtual environment exists at /home/ubuntu/particle-agent/.venv.
Install service:
sudo cp particle.service /etc/systemd/system/particle.service
sudo systemctl daemon-reload
sudo systemctl enable --now particle.service
Inspect logs:
journalctl -u particle.service -f
Project Layout
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
