# Jarvis Backend

## Schnellstart mit Docker Compose
1. **Umgebungsdatei erstellen:**
   ```bash
   cp .env.example .env
   # Trage mindestens deinen Home-Assistant-Token unter HA_TOKEN ein
   ```
2. **Container starten:**
   ```bash
   docker compose up --build
   ```
   * Der Backend-Service läuft anschließend auf http://localhost:8000.
   * Die Wyoming-Services (Whisper, Piper, OpenWakeWord) werden automatisch mitgestartet.

## Einzelnen Backend-Container bauen und starten
Falls du nur den FastAPI-Container ohne die zusätzlichen Wyoming-Services starten möchtest:
```bash
docker build -t jarvis-backend .
docker run --env-file .env -p 8000:8000 jarvis-backend
```

## Wichtige Hinweise
- Stelle sicher, dass Docker und Docker Compose installiert sind.
- Alle relevanten Umgebungsvariablen und Standardwerte findest du in [.env.example](.env.example).
- Für den Zugriff auf Home Assistant muss `HA_URL` erreichbar sein und `HA_TOKEN` gültig gesetzt sein.
