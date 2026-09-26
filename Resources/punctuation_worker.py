"""Offline FullStop worker. Only inserts punctuation; never generates words."""
import json
import os
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def protected(word):
    # Never change URLs, filenames, addresses, paths, numbers or code tokens.
    return bool(re.search(r"[\d@/_=\\]", word) or re.search(r"\w[.:]\w", word))


def apply_labels(text, predictions):
    matches = list(re.finditer(r"\S+", text))
    if len(matches) != len(predictions):
        return text
    chunks, cursor = [], 0
    for match, (label, score) in zip(matches, predictions):
        word = match.group()
        chunks.append(text[cursor:match.end()])
        if (label in {",", ".", "?", ":"} and score >= 0.75
                and not protected(word) and word[-1].isalpha()):
            chunks.append(label)
        cursor = match.end()
    chunks.append(text[cursor:])
    return "".join(chunks)


class Punctuator:
    def __init__(self, path, device="cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        torch.set_num_threads(4)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForTokenClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).eval().to(device)

    def clean(self, text):
        words = text.split()
        if not words or len(text) > 16000:
            return text
        batch = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=False)
        if batch["input_ids"].shape[1] > 480:
            return text  # No silent truncation and no context-breaking chunking.
        word_ids = batch.word_ids()
        with self.torch.inference_mode():
            logits = self.model(**{k:v.to(self.device) for k,v in batch.items()}).logits[0]
            probabilities = logits.softmax(-1).cpu()
        # Match upstream alignment: the last subtoken determines a word's label.
        labels = [("0", 1.0)] * len(words)
        for i, wid in enumerate(word_ids):
            if wid is not None:
                score, label = probabilities[i].max(-1)
                labels[wid] = (self.model.config.id2label[int(label)], float(score))
        return apply_labels(text, labels)


def main():
    model = Punctuator(sys.argv[1], os.environ.get("FLOW_PUNCTUATION_DEVICE", "cpu"))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            item = json.loads(line)
            text = item["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            started = time.perf_counter()
            result = model.clean(text)
            print(json.dumps({"text": result, "seconds": time.perf_counter()-started}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__}), flush=True)


if __name__ == "__main__":
    main()
