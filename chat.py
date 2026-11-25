#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path

APP_NAME = "Glow Up Ai"
DEVELOPER = "Adam xit"
HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SYSTEM_FILE = HERE / "system_prompt.txt"
HISTORY_FILE = HERE / "history.jsonl"
LLAMA_CLI = "/data/data/com.termux/files/usr/bin/llama-cli"
MAX_TURNS = 3
CTX_SIZE = 1024
MAX_TOKENS = 120
THREADS = 4

NAME_PATTERNS = [
    r"what(?:'s| is) your name",
    r"what do (?:they|people) call you",
    r"ما اسمك",
    r"ما هو اسمك",
    r"إيش اسمك",
    r"شنو اسمك",
]
DEV_PATTERNS = [
    r"who (?:developed|made|created|built) you",
    r"who is your (?:developer|creator|maker)",
    r"من صنعك",
    r"من طورك",
    r"من أنشأك",
    r"مين طورك",
]
IDENTITY_PATTERNS = [
    r"who are you",
    r"من أنت",
    r"مين انت",
]

ANSWER_NAME = "Glow Up Ai"
ANSWER_DEVELOPER = "Adam xit"
ANSWER_IDENTITY = "I am Glow Up Ai, developed by Adam xit."


def matches(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def identity_answer(text):
    if matches(DEV_PATTERNS, text):
        return ANSWER_DEVELOPER
    if matches(NAME_PATTERNS, text):
        return ANSWER_NAME
    if matches(IDENTITY_PATTERNS, text):
        return ANSWER_IDENTITY
    return None


def load_history():
    if not HISTORY_FILE.exists():
        return []
    items = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                items.append(entry)
        except (ValueError, TypeError):
            continue
    return items


def save_turn(role, content):
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")


def clear_history():
    HISTORY_FILE.write_text("", encoding="utf-8")


def build_prompt(history, user_text):
    system = SYSTEM_FILE.read_text(encoding="utf-8").strip()
    parts = ["<|im_start|>system", system, "<|im_end|>"]
    for entry in history[-MAX_TURNS * 2:]:
        parts.extend(["<|im_start|>" + entry["role"], entry["content"], "<|im_end|>"])
    parts.extend(["<|im_start|>user", user_text, "<|im_end|>", "<|im_start|>assistant"])
    return "\n".join(parts) + "\n"


def model_answer(prompt):
    command = [
        LLAMA_CLI,
        "-m", str(MODEL),
        "-p", prompt,
        "-n", str(MAX_TOKENS),
        "-c", str(CTX_SIZE),
        "-t", str(THREADS),
        "--temp", "0.7",
        "--repeat-penalty", "1.05",
        "--no-warmup",
        "--log-disable",
        "-st",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "Error: the model timed out after 300 seconds."
    except OSError as exc:
        return "Error: could not start llama-cli: " + str(exc)
    if result.returncode != 0:
        return "Error (llama-cli exit " + str(result.returncode) + "): " + (result.stderr or result.stdout).strip()
    tail = result.stdout
    candidates = []
    for marker in ("<|im_start|>assistant", "(truncated)"):
        index = tail.rfind(marker)
        if index != -1:
            candidates.append((index, len(marker)))
    if candidates:
        index, length = max(candidates)
        tail = tail[index + length:]
    else:
        error_index = tail.find("Error:")
        if error_index != -1:
            tail = tail[error_index:]
    tail = re.split(r"\[ Prompt:|Exiting\.\.\.|\[end of text\]", tail)[0]
    return tail.strip()


def respond(user_text, history):
    fixed = identity_answer(user_text)
    if fixed is not None:
        return fixed
    return model_answer(build_prompt(history, user_text))


def banner():
    sys.stdout.write("\033]0;" + APP_NAME + "\007")
    print("=" * 46)
    print("  " + APP_NAME)
    print("  Developed by " + DEVELOPER)
    print("  Fully offline - local model, no telemetry")
    print("=" * 46)
    print("Type /exit to quit, /clear to reset history, /about for info.")


def main():
    if not MODEL.exists():
        print("Error: model not found at " + str(MODEL))
        return 1
    if not SYSTEM_FILE.exists():
        print("Error: system prompt not found at " + str(SYSTEM_FILE))
        return 1
    history = load_history()
    banner()
    if history:
        print("Loaded " + str(len(history)) + " saved messages.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye.")
            break
        if user_text.lower() == "/clear":
            clear_history()
            history = []
            print("History cleared.")
            continue
        if user_text.lower() == "/about":
            print(APP_NAME + " - Developed by " + DEVELOPER + " - Fully offline")
            continue
        answer = respond(user_text, history)
        print(APP_NAME + ": " + answer)
        if not answer.startswith("Error:"):
            history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
            save_turn("user", user_text)
            save_turn("assistant", answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
