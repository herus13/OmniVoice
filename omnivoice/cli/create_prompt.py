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

"""Create and save a reusable voice clone prompt.

Encodes a voice into a portable prompt file once, so later synthesis can
reuse it without re-running the audio tokenizer (and ASR) every time.

Two ways to create the voice:

  * Clone an existing recording with ``--ref_audio`` (+ optional
    ``--ref_text``).
  * Design a new voice with ``--instruct`` (voice-design attributes). A
    short sample is rendered with that instruct and then captured as a
    clone prompt. Because voice design samples a random voice each call,
    ``--seed`` selects WHICH voice is locked — a random seed is chosen and
    logged if you omit it. The saved prompt is a fixed clone prompt: the
    instruct/seed are used once to capture the voice and cannot be
    re-rendered with different design settings afterwards.

Every run writes two files: the prompt (``--output``, e.g. ``narrator.pt``)
and a review WAV alongside it (``narrator.wav`` by default, or
``--sample_output``) so you can listen to the captured voice.

Note: this command must load the model (encoding/rendering needs it), so
the first run downloads the multi-GB checkpoint (cached afterwards).

Usage:
    # Clone an existing voice  -> voice.pt + voice.wav
    omnivoice-create-prompt --ref_audio voice.wav --ref_text "..." \
        --output voice.pt

    # Design a voice and save it (seed locks the identity)
    #   -> narrator.pt + narrator.wav
    omnivoice-create-prompt \
        --instruct "male, middle-aged, moderate pitch, british accent" \
        --seed 7 --output narrator.pt

    # Reuse it later, no reference needed
    omnivoice-infer --voice_clone_prompt narrator.pt \
        --text "Chapter one." --output out.wav
"""

import argparse
import logging
import random
from pathlib import Path

import torch

import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device, str2bool

# Phoneme-rich ~12s passage used to render a designed voice before capture.
DEFAULT_SAMPLE_TEXT = (
    "The north wind and the sun disputed which was the stronger, when a "
    "traveler came along wrapped in a warm cloak. They agreed that whoever "
    "first made the traveler take off his cloak should be the winner."
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reusable OmniVoice voice clone prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    # Clone mode
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Clone mode: reference audio file path to clone.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Clone mode: transcript of --ref_audio. Auto-transcribed via "
        "ASR if omitted.",
    )
    # Design mode
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Design mode: voice-design attributes (e.g. 'male, middle-aged, "
        "moderate pitch, british accent'). Mutually exclusive with --ref_audio.",
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default=DEFAULT_SAMPLE_TEXT,
        help="Design mode: text rendered to capture the designed voice. "
        "Override for non-English voices. Default: a ~12s English passage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Design mode: seed selecting WHICH random design voice is "
        "locked. A random seed is chosen and logged if omitted.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Design mode: language name/code for the sample text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output prompt file path (conventionally .pt).",
    )
    parser.add_argument(
        "--sample_output",
        type=str,
        default=None,
        help="Path for a review WAV of the captured voice. Defaults to "
        "--output with a .wav extension. In design mode this is the rendered "
        "sample; in clone mode it is the captured reference decoded back.",
    )
    parser.add_argument(
        "--preprocess_prompt",
        type=str2bool,
        default=True,
        help="Apply silence removal/trimming to the reference audio and add "
        "trailing punctuation to the transcript.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Auto-detected if not specified.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    # Exactly one of clone (--ref_audio) or design (--instruct).
    if bool(args.ref_audio) == bool(args.instruct):
        raise SystemExit(
            "Provide exactly one of --ref_audio (clone) or --instruct (design)."
        )
    if args.instruct and args.ref_text:
        logging.warning("--ref_text is ignored in design mode; use --sample_text.")

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    if args.ref_audio:
        logging.info(f"Encoding reference audio: {args.ref_audio}")
        prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            preprocess_prompt=args.preprocess_prompt,
        )
        # Decode the captured tokens so the review WAV is exactly what the
        # prompt holds (post silence-trim / normalisation).
        review_wav = (
            model.audio_tokenizer.decode(
                prompt.ref_audio_tokens.unsqueeze(0).to(model.audio_tokenizer.device)
            )
            .audio_values[0]
            .cpu()
            .numpy()
            .squeeze()
        )
    else:
        seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        logging.info(
            f"Designing voice (instruct={args.instruct!r}, seed={seed}) ..."
        )
        audios = model.generate(
            text=args.sample_text,
            instruct=args.instruct,
            language=args.language,
            seed=seed,
        )
        review_wav = audios[0]
        sample_wav = torch.from_numpy(review_wav)
        prompt = model.create_voice_clone_prompt(
            ref_audio=(sample_wav, model.sampling_rate),
            ref_text=args.sample_text,
            preprocess_prompt=args.preprocess_prompt,
        )
        logging.info(f"Locked designed voice with seed={seed}.")

    prompt.save(args.output, model=args.model)
    logging.info(
        f"Saved voice clone prompt to {args.output} "
        f"(ref_text: {prompt.ref_text[:60]})"
    )

    sample_output = args.sample_output or str(Path(args.output).with_suffix(".wav"))
    sf.write(sample_output, review_wav, model.sampling_rate)
    logging.info(f"Saved review audio to {sample_output}")


if __name__ == "__main__":
    main()
