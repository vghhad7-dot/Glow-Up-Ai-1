# Glow Up Ai

Developed by Adam xit

A fully offline AI chat assistant for Android. All inference happens on the phone with llama.cpp. No external API calls, no telemetry, no account.

## Android APK

The Android app is in `android-app/`. It runs the same Qwen2.5 0.5B Q4_K_M GGUF model through a small JNI bridge; it does not start `llama-cli` or any other binary.

GitHub Actions workflow `Build Glow Up Ai debug APK` builds the app:

- Checks out the repository with Git LFS enabled, so `models/*.gguf` is present.
- Installs JDK 17, Android SDK 34, NDK 26.3.11579264 and CMake 3.22.1.
- Builds `android-app` with Gradle 8.7 and AGP 8.5.2.
- Uploads the debug APK as the `GlowUpAi-debug` artifact.

To install: download the `GlowUpAi-debug` artifact from the successful workflow run, unzip it, copy `debug.apk` to the phone, allow installation from your file manager when Android asks, then open it. On first launch the app copies the bundled model into its private `files/models/` folder and then works in airplane mode.

## Termux version

The original Termux/Python version is still supported.

- Runtime: llama.cpp (`llama-cli`) installed by Termux package manager, kept outside shared storage in `/data/data/com.termux/files/usr`.
- Project, model, and history: `/storage/emulated/0/Download/glow-Up-Ai/`
- Default model: Qwen2.5 0.5B Instruct Q4_K_M (multilingual, Arabic and English).

A 3B/7B Q4 model does not fit this phone's ~0.8 GiB free RAM, so the 0.5B Q4 model is used. It still supports Arabic and English.

### Install

Everything is already installed. If you need to rebuild on another phone:

```bash
pkg install llama-cpp python
mkdir -p /storage/emulated/0/Download/glow-Up-Ai/models
curl -L -o /storage/emulated/0/Download/glow-Up-Ai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

If `/storage/emulated/0` is not visible, run `termux-setup-storage` first.

### Run

Shared storage cannot execute files, so run the script with bash:

```bash
bash /storage/emulated/0/Download/glow-Up-Ai/start.sh
```

After the model has been downloaded, the assistant works in airplane mode.

Commands inside chat: `/exit`, `/clear`, `/about`.

### Identity

The system prompt in `system_prompt.txt` sets the name to Glow Up Ai and the developer to Adam xit. Identity questions are also answered by an exact local rule so the answer never drifts:

- "what's your name?" -> `Glow Up Ai`
- "who developed you?" -> `Adam xit`
- "who are you?" -> `I am Glow Up Ai, developed by Adam xit.`

### History

Conversation turns are appended to `history.jsonl` inside the project folder. Nothing is sent anywhere. Use `/clear` to empty it.

### Switch models

1. Download another GGUF into the `models/` folder, for example:

```bash
curl -L -o /storage/emulated/0/Download/glow-Up-Ai/models/MODEL.gguf URL
```

2. Edit `chat.py` and change the `MODEL` line to the new filename.

3. For a different size, also adjust `CTX_SIZE`, `MAX_TOKENS`, and `THREADS` near the top of `chat.py`. Keep the model smaller than available RAM.

Recommended alternatives that fit this phone: `qwen2.5-0.5b-instruct-q3_k_m.gguf`, `qwen2.5-0.5b-instruct-q4_0.gguf`.

### Offline check

The chat only starts a local `llama-cli` process and reads local files. It imports no networking library. A test run can be done with all non-loopback sockets blocked; the assistant still answers.
