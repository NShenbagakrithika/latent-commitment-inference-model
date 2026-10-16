#!/usr/bin/env python3
"""Test the saved V4 workplace commitment model locally, without training."""
import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

# Inference uses only the bundled files; no Hugging Face downloads or telemetry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent


class CommitmentModel:
    def __init__(self):
        # Transformers 4.57 uses this legacy decorator; it is harmless for inference.
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        model_dir = BASE_DIR / "best_model"
        expected = json.loads((BASE_DIR / "model_integrity.json").read_text())
        for filename, checksum in expected.items():
            path = model_dir / filename
            if not path.is_file():
                raise RuntimeError(f"Missing model file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != checksum:
                raise RuntimeError(f"Model verification failed: {filename}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=False, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to("cpu").eval()

    def predict(self, context, reply):
        if not isinstance(context, str) or not isinstance(reply, str):
            raise ValueError("Context and employee reply must be text.")
        if not context.strip() or not reply.strip():
            raise ValueError("Enter both a context and an employee reply.")
        text = (
            f"[CONTEXT] {context.strip()} [TARGET SPEAKER] employee "
            f"[TARGET UTTERANCE] {reply.strip()}"
        )
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        length = tokens["input_ids"].shape[1]
        if length > 256:
            raise ValueError(
                f"Input has {length} tokens; the limit is 256. "
                "Shorten the earlier context. No text was silently truncated."
            )
        with self.torch.inference_mode():
            probabilities = self.model(**tokens).logits.float().softmax(-1)[0].tolist()
        category = max(range(len(probabilities)), key=probabilities.__getitem__)
        labels = self.model.config.id2label
        return {
            "label": labels[category],
            "label_id": category,
            "score": probabilities[category],
            "tokens": length,
            "probabilities": {labels[i]: value for i, value in enumerate(probabilities)},
        }


def print_result(result, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print("\nPrediction:", result["label"].replace("_", " "))
    for label, score in result["probabilities"].items():
        print(f"  {label.replace('_', ' '):24s} {score:7.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", help="Earlier messages, including speaker labels.")
    parser.add_argument("--reply", help="The employee's new reply to classify.")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    args = parser.parse_args()
    if (args.context is None) != (args.reply is None):
        parser.error("Provide both --context and --reply, or neither for interactive mode.")

    print("Loading the verified V4 model on CPU...", file=sys.stderr)
    try:
        runner = CommitmentModel()
    except Exception as error:
        parser.exit(1, f"Could not load the saved model: {error}\n")

    if args.context is not None:
        try:
            print_result(runner.predict(args.context, args.reply), args.json)
        except ValueError as error:
            parser.exit(2, f"Input error: {error}\n")
        return

    print("Ready. Predictions run offline; no training is performed.")
    print("Scores are uncalibrated. Review ambiguous conversations.")
    print("Paste the earlier messages on one line, using speaker labels such as [manager]:.")
    print("Type quit at the Context prompt to exit. Ctrl+C also exits.")
    while True:
        try:
            context = input("\nContext: ").strip()
            if context.lower() in {"quit", "exit", "q"}:
                break
            if not context:
                print("Please enter the earlier conversation.")
                continue
            reply = input("Employee reply: ")
            print_result(runner.predict(context, reply), args.json)
        except ValueError as error:
            print("Input error:", error)
        except (EOFError, KeyboardInterrupt):
            print()
            break
    print("Goodbye.")


if __name__ == "__main__":
    main()
