#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-speaker dialogue inference CLI for OmniVoice.

Synthesize a conversation between two or more speakers into a single
stitched WAV. Each speaker keeps a fixed voice across all their turns.

Usage:
    omnivoice-dialogue --model k2-fsa/OmniVoice \
        --script script.jsonl --speakers speakers.json --output convo.wav

``--script`` is a JSONL file, one ordered turn per line:
    {"speaker": "A", "text": "Hi there!"}
    {"speaker": "B", "text": "Hello, how are you?"}

``--speakers`` is a JSON file mapping each speaker id to a voice spec.
Each speaker has exactly one of: ``ref_audio`` (+ optional ``ref_text``)
for voice clone, ``instruct`` for voice design, or ``voice_clone_prompt``
(a path to a saved prompt from ``omnivoice-create-prompt``, reused without
re-encoding). ``language`` is optional:
    {
      "A": {"ref_audio": "male.wav", "ref_text": "..."},
      "B": {"instruct": "female, young adult", "language": "English"},
      "C": {"voice_clone_prompt": "narrator.pt"}
    }

Turns are generated independently and concatenated with a fixed silence
gap. Per-turn voice is consistent (same speaker spec each turn); with
``--seed`` set, a turn is reproducible per generate() call (same speaker +
same text + same seed -> identical audio).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.common import get_best_device

# Fixed silence inserted between turns (seconds). Intra-turn chunk stitching
# is handled separately by cross_fade_chunks inside generate().
INTER_TURN_GAP_S = 0.3


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice multi-speaker dialogue inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="JSONL file of ordered turns; each line {\"speaker\": id, "
        "\"text\": str}.",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        required=True,
        help="JSON file mapping speaker id -> voice spec. Each spec has "
        "exactly one of 'ref_audio' (voice clone), 'instruct' (voice "
        "design), or 'voice_clone_prompt' (saved prompt path); optional "
        "'ref_text' and 'language'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path for the stitched conversation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output. Applied per generate() "
        "call (per-turn): same speaker + text + seed -> identical audio. "
        "Omit for random output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def _load_speakers(path: str) -> dict:
    """Load and validate the speakers map. Returns {id: spec}."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--speakers file not found: {path}")
    try:
        speakers = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"--speakers is not valid JSON: {e}")
    if not isinstance(speakers, dict) or not speakers:
        raise SystemExit("--speakers must be a non-empty JSON object.")

    for spk, spec in speakers.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"Speaker '{spk}' spec must be a JSON object.")
        modes = [
            k
            for k in ("ref_audio", "instruct", "voice_clone_prompt")
            if spec.get(k)
        ]
        if len(modes) != 1:
            raise SystemExit(
                f"Speaker '{spk}' must have exactly one of 'ref_audio' (voice "
                "clone), 'instruct' (voice design), or 'voice_clone_prompt' "
                "(saved prompt) — got "
                + (", ".join(modes) if modes else "none")
                + "."
            )
        for key in ("ref_audio", "voice_clone_prompt"):
            if spec.get(key) and not Path(spec[key]).is_file():
                raise SystemExit(f"Speaker '{spk}' {key} not found: {spec[key]}")
    return speakers


def _load_script(path: str, speakers: dict) -> list:
    """Load and validate ordered turns. Returns [(speaker, text), ...]."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--script file not found: {path}")

    turns = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--script line {line_no} is not valid JSON: {e}")
            spk = obj.get("speaker")
            text = obj.get("text")
            if not spk or not text or not str(text).strip():
                raise SystemExit(
                    f"--script line {line_no} needs non-empty 'speaker' and "
                    "'text'."
                )
            if spk not in speakers:
                raise SystemExit(
                    f"--script line {line_no}: speaker '{spk}' not defined in "
                    "--speakers."
                )
            turns.append((spk, str(text).strip()))

    if not turns:
        raise SystemExit("--script contains no turns; nothing to synthesize.")
    return turns


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    # Validate both inputs BEFORE loading the model (multi-GB download).
    speakers = _load_speakers(args.speakers)
    turns = _load_script(args.script, speakers)
    logging.info(
        f"Loaded {len(turns)} turns across {len(speakers)} speakers."
    )

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    # Resolve a reusable voice-clone prompt once per clone speaker, so the
    # reference is encoded a single time rather than per turn. Saved prompts
    # are loaded directly (no re-encode); raw ref_audio is encoded now.
    clone_prompts = {}
    for spk, spec in speakers.items():
        if spec.get("voice_clone_prompt"):
            logging.info(
                f"Loading saved voice clone prompt for speaker '{spk}' "
                f"from {spec['voice_clone_prompt']} ..."
            )
            clone_prompts[spk] = VoiceClonePrompt.load(spec["voice_clone_prompt"])
        elif spec.get("ref_audio"):
            logging.info(f"Building voice clone prompt for speaker '{spk}' ...")
            clone_prompts[spk] = model.create_voice_clone_prompt(
                ref_audio=spec["ref_audio"],
                ref_text=spec.get("ref_text"),
            )

    gap = np.zeros(int(INTER_TURN_GAP_S * model.sampling_rate), dtype=np.float32)

    segments = []
    for i, (spk, text) in enumerate(turns):
        spec = speakers[spk]
        kw = dict(text=text, language=spec.get("language"), seed=args.seed)
        if spk in clone_prompts:
            kw["voice_clone_prompt"] = clone_prompts[spk]
        else:
            kw["instruct"] = spec["instruct"]

        logging.info(f"Turn {i + 1}/{len(turns)} [{spk}]: {text[:60]}...")
        audio = model.generate(**kw)[0]

        if segments:
            segments.append(gap)
        segments.append(audio.astype(np.float32))

    conversation = np.concatenate(segments)
    sf.write(args.output, conversation, model.sampling_rate)
    duration = conversation.shape[-1] / model.sampling_rate
    logging.info(f"Saved {duration:.1f}s conversation to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
