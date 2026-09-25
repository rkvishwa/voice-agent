
# Agent Instructions: Hybrid Full-Duplex Voice Agent

## 1. Project Goal
Build a low-latency, full-duplex, browser-accessible voice assistant deployed on a Linux VPS without dedicated GPUs.
The agent must:
1. Accept bidirectional streaming audio over WebSockets (16 kHz 16-bit PCM).
2. Transcribe incoming user audio locally on CPU using **Faster-Whisper** (`base.en` with CTranslate2 int8 quantization).
3. Offload reasoning and generation to an **Azure OpenAI / Foundry** endpoint (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`) using streaming tokens.
4. Synthesize spoken responses locally on CPU using **Kokoro-82M ONNX** (`kokoro-onnx`) chunked on sentence boundaries.
5. Provide instant client/server **barge-in interruption**: when the user speaks over the assistant, the server cancels active LLM/TTS generation tasks and signals the browser to purge pending audio playback buffers immediately.

---

## 2. Target Architecture & Stack


```

[ Client Browser ]
│  ▲
│  │  Bidirectional Audio via WebSocket (PCM16 / WAV bytes)
▼  │
[ FastAPI / Uvicorn Server ]
├── Energy-based VAD / Barge-In Interrupt Handler
├── Faster-Whisper (CPU, int8, base.en) ──► Produces text prompt
├── AsyncAzureOpenAI Client ──────────────► Streams text tokens
└── Kokoro-82M ONNX (CPU, StyleTTS 2) ────► Streams WAV chunks back

```

* **Backend:** Python 3.11+, FastAPI, Uvicorn, WebSockets.
* **Inference Libraries:** `faster-whisper`, `kokoro-onnx`, `soundfile`, `numpy`.
* **LLM Client:** `openai` (`AsyncAzureOpenAI`).
* **Environment:** Systemd service on Ubuntu 22.04 / 24.04 LTS.

---

## 3. Directory & File Structure to Generate

Maintain this strict directory layout:

```text
/opt/voice-agent/ (or project root)
├── .env
├── requirements.txt
├── models/
│   ├── kokoro-v0_19.onnx
│   └── voices.bin
├── app/
│   ├── __init__.py
│   ├── config.py             # Parses env vars (Azure credentials, ports, paths)
│   ├── vad.py                # Audio energy & pause detection logic
│   ├── asr.py                # Faster-Whisper wrapper
│   ├── tts.py                # Kokoro-82M wrapper with dynamic signature inspection
│   ├── llm.py                # Async Azure OpenAI streaming client
│   ├── server.py             # FastAPI app & WebSocket orchestration loop
│   └── static/
│       └── index.html        # WebRTC/WebSocket browser test console
├── scripts/
│   ├── download_models.sh    # Script to fetch Kokoro ONNX assets
│   └── setup_systemd.sh      # Service registration script
└── run.sh                    # Entrypoint execution script

```

---

## 4. Implementation Requirements

### A. Environment Configuration (`app/config.py`)

Load via `python-dotenv`:

* `AZURE_OPENAI_ENDPOINT` (Required)
* `AZURE_OPENAI_API_KEY` (Required)
* `AZURE_OPENAI_DEPLOYMENT` (Required)
* `AZURE_OPENAI_API_VERSION` (Default: `"2024-06-01"`)
* `HOST` (Default: `"0.0.0.0"`)
* `PORT` (Default: `8000`)
* `MODELS_DIR` (Default: `/opt/voice-agent/models`)

Fail fast with an informative error if mandatory Azure credentials are missing.

### B. STT Module (`app/asr.py`)

* Use `faster_whisper.WhisperModel`.
* Instantiate with `device="cpu"`, `compute_type="int8"`, `cpu_threads=2`.
* Model size default: `base.en`.
* Method: `transcribe(audio_float32_array, beam_size=1, language="en")`.

### C. TTS Module (`app/tts.py`)

* Load `kokoro_onnx.Kokoro` pointing to `models/kokoro-v0_19.onnx` and `models/voices.bin`.
* Use voice `af_heart` (or fallback to `am_adam`), `speed=1.05`.
* Must inspect `inspect.signature(kokoro.create)` at runtime to supply `lang="en-us"` safely if supported by the installed library version.
* Convert raw audio samples into WAV bytes in-memory using `io.BytesIO` and `soundfile.write(..., format='WAV')`.

### D. LLM Module (`app/llm.py`)

* Initialize `AsyncAzureOpenAI` using configuration variables.
* Function `stream_llm_response(history: list[dict]) -> AsyncGenerator[str, None]`:
* Sets `max_tokens=100`, `temperature=0.7`.
* Enforces a concise system prompt: `"You are a natural, concise conversational voice assistant. Limit answers to 1-2 spoken sentences."`
* Yields text deltas token-by-token.



### E. Barge-In & Audio Processing Loop (`app/server.py`)

1. Listen for raw binary 16 kHz 16-bit PCM frames from the client.
2. Calculate RMS energy per frame:

$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$$


3. **Barge-In Detection:**
* If $\text{RMS} > 0.035$ while an AI turn task is actively processing or speaking:
* Cancel the active `asyncio.Task`.
* Send `{"type": "barge_in"}` JSON packet to the client so the browser immediately stops playing audio.




4. **Turn-End Detection:**
* Accumulate frames into a buffer while in speech.
* If silence persists for $> 6$ frames (~500–600 ms) and buffer length $> 0.4$ seconds:
* Launch a background task `process_turn(audio_buffer)`.




5. **Punctuation-based Streaming:**
* As tokens arrive from the LLM, accumulate into a sub-sentence buffer.
* Split on delimiters (`.`, `,`, `!`, `?`, `\n`).
* Dispatch the sub-sentence to `tts.synth_chunk()` and transmit binary WAV bytes over the WebSocket immediately.
* Send JSON transcript updates for UI inspection (`{"type": "transcript", "role": "user"|"agent", "text": "..."}`).



### F. Browser Interface (`app/static/index.html`)

* Pure vanilla HTML5 + JavaScript (no complex build steps or node packages).
* Microphone capture using `AudioContext` (16 kHz, 1-channel) + `ScriptProcessorNode` / `AudioWorkletNode`.
* Convert incoming Float32 mic input to Int16 PCM before sending.
* Implement an audio queue for incoming binary WAV chunks:
* When a chunk arrives, decode and push to queue.
* When `barge_in` JSON is received, immediately call `currentSource.stop()`, clear `queue = []`, and reset playback state.



---

## 5. Deployment & Systemd Automation (`scripts/setup_systemd.sh`)

* Create a systemd unit at `/etc/systemd/system/voice-agent.service`.
* Read environment variables from `/opt/voice-agent/.env`.
* Execute via virtual environment: `/opt/voice-agent/venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000`.
* Enable and start the service with auto-restart on crash (`Restart=always`, `RestartSec=3`).

---

## 6. Development Rules & Guardrails for the Agent

1. **Never use heavy PyTorch Whisper:** Always use `faster-whisper` (CTranslate2) with `int8` quantization to avoid CPU starvation.
2. **Never buffer full LLM replies before TTS:** The pipeline MUST stream and split on clauses/commas; waiting for complete LLM completion will ruin voice turnaround latency.
3. **No blocking operations:** All I/O (Azure API, WebSocket sends) must be strictly asynchronous. Wrap CPU-bound inference (`whisper.transcribe`, `kokoro.create`) in `asyncio.to_thread()` where appropriate to prevent event-loop freezing.
4. **Self-contained frontend:** Do not introduce external frontend bundlers (React, Vite, Webpack). Deliver the web interface via a single static or templated HTML file.
