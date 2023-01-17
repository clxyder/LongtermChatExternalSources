import os
import json
from time import time

import numpy as np

def open_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath: str, content: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath: str, payload: dict) -> None:
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def log_json_message(info: dict, speaker: str, chat_log_dir: str) -> None:
    filename = f'log_{time()}_{speaker}.json'
    save_location = os.path.join(chat_log_dir, filename)
    save_json(save_location, info)


def cosine_similarity(a, b):
    # from: openai.embeddings_utils.cosine_similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
