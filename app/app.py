import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = os.environ.get("OCR_MODEL_PATH", "/models/qwen")
OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR", "/app/output"))
SOCKET_PATH = "/tmp/ocr_model.sock"

PROMPT = """Read the target text in this image.

For vehicle registration plates:
- Return only the registration characters.
- Exclude jurisdiction banners and slogans.
- Keep Chinese province characters and registration letters.

For signs:
- Transcribe all visible target sign text in reading order.
- Read lines from top to bottom and join them with a single space.
- Exclude unrelated background text.
- Do not invent unreadable text.

Return only the transcription. Do not add explanations or labels."""


def load_model():
    print("Loading OCR model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    print("Model loaded and ready.", flush=True)
    return processor, model


def process_image(image_path, processor, model):
    image_path = Path(image_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_output.json"

    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    response = {"text": result, "confidence": 0.90}
    output_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return response


def worker():
    # This process owns the model and keeps it loaded between images.
    processor, model = load_model()
    try:
        Image.new("RGB", (640, 200), "white").save("/tmp/warmup.png")
        process_image("/tmp/warmup.png", processor, model)
        (OUTPUT_DIR / "warmup_output.json").unlink(missing_ok=True)
        print("Warm-up done.", flush=True)
    except Exception as exc:
        print("Warm-up failed:", exc, flush=True)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(SOCKET_PATH)
        server.listen(4)
        print("OCR worker listening.", flush=True)

        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = json.loads(
                        conn.makefile("r", encoding="utf-8").readline()
                    )
                    result = process_image(
                        request["input_image"], processor, model
                    )
                    reply = {"ok": True, "result": result}
                except Exception as exc:
                    reply = {"ok": False, "error": str(exc)}

                conn.sendall(
                    (json.dumps(reply, ensure_ascii=False) + "\n").encode()
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        worker()
        return

    if not args.input_image:
        parser.error("--input-image is required")

    # Wait for the model worker to finish loading.
    deadline = time.monotonic() + 600
    while True:
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(600)
            client.connect(SOCKET_PATH)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                raise RuntimeError("OCR worker did not become ready in time")
            time.sleep(1)

    with client:
        client.sendall(
            (json.dumps({"input_image": args.input_image}) + "\n").encode()
        )
        with client.makefile("r", encoding="utf-8") as stream:
            reply = json.loads(stream.readline())

    if not reply["ok"]:
        raise RuntimeError(reply["error"])

    print(f"Text: {reply['result']['text']}", flush=True)


if __name__ == "__main__":
    main()
