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
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.dialogue import INTER_TURN_GAP_S
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# Dropdown sentinel for "no saved voice selected".
_NO_SAVED_VOICE = "(none)"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--prompts-dir",
        default="omnivoice_voices",
        help="Directory where saved voice clone prompts (.pt) are stored and "
        "listed in the Voice Clone tab (default: ./omnivoice_voices).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    prompts_dir: str = "omnivoice_voices",
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    prompts_path = Path(prompts_dir)
    prompts_path.mkdir(parents=True, exist_ok=True)

    def _list_prompts():
        return sorted(p.stem for p in prompts_path.glob("*.pt"))

    def _saved_choices():
        return [_NO_SAVED_VOICE] + _list_prompts()

    def _sanitize_name(name: str) -> str:
        safe = "".join(
            c for c in (name or "") if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        return safe.replace(" ", "_")

    def _save_voice(ref_audio, ref_text, name):
        if not ref_audio:
            return "Upload a reference audio first.", gr.update()
        safe = _sanitize_name(name)
        if not safe:
            return "Enter a valid name for the voice.", gr.update()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=(ref_text or None)
            )
            prompt.save(str(prompts_path / f"{safe}.pt"), model=checkpoint)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}", gr.update()
        return (
            f"Saved voice '{safe}'.",
            gr.update(choices=_saved_choices(), value=safe),
        )

    def _save_design_voice(audio, text, name):
        # Capture the voice the user just heard (no re-generation, which
        # would sample a different random design voice).
        if audio is None:
            return "Generate a voice first, then save.", gr.update()
        safe = _sanitize_name(name)
        if not safe:
            return "Enter a valid name for the voice.", gr.update()
        sr, arr = audio
        wav = np.asarray(arr).astype(np.float32) / 32767.0
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(wav), int(sr)),
                ref_text=(text.strip() if text and text.strip() else None),
            )
            prompt.save(str(prompts_path / f"{safe}.pt"), model=checkpoint)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}", gr.update()
        return (
            f"Saved designed voice '{safe}'. Pick it in the Voice Clone tab.",
            gr.update(choices=_saved_choices(), value=safe),
        )

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        silence_duration,
        seed,
        mode,
        ref_text=None,
        saved_voice=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            silence_duration=(
                float(silence_duration) if silence_duration is not None else 0.3
            ),
            seed=int(seed) if seed is not None else None,
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if saved_voice and saved_voice != _NO_SAVED_VOICE:
                path = prompts_path / f"{saved_voice}.pt"
                if not path.is_file():
                    return None, f"Saved voice not found: {saved_voice}"
                kw["voice_clone_prompt"] = VoiceClonePrompt.load(str(path))
            elif ref_audio:
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            else:
                return None, "Upload a reference audio or pick a saved voice."

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # -- multi-speaker dialogue core --
    def _run_dialogue(
        a_audio,
        a_ref_text,
        a_instruct,
        b_audio,
        b_ref_text,
        b_instruct,
        script,
        language,
        seed,
    ):
        if not script or not script.strip():
            return None, "Please enter a dialogue script."

        # Parse "A: text" / "B: text" lines into ordered turns.
        turns = []
        for ln_no, raw in enumerate(script.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                return None, f"Line {ln_no}: expected 'A: text' or 'B: text'."
            label, text = line.split(":", 1)
            label, text = label.strip().upper(), text.strip()
            if label not in ("A", "B"):
                return None, f"Line {ln_no}: speaker '{label}' must be A or B."
            if not text:
                return None, f"Line {ln_no}: empty text."
            turns.append((label, text))

        if not turns:
            return None, "No turns parsed from script."

        # Configure a voice for each label actually used (clone if reference
        # audio given, else voice design via instruct).
        cfg = {
            "A": (a_audio, a_ref_text, a_instruct),
            "B": (b_audio, b_ref_text, b_instruct),
        }
        voices = {}
        for label in {lbl for lbl, _ in turns}:
            ref_aud, ref_txt, instr = cfg[label]
            if ref_aud:
                voices[label] = (
                    "clone",
                    model.create_voice_clone_prompt(
                        ref_audio=ref_aud, ref_text=ref_txt or None
                    ),
                )
            elif instr and instr.strip():
                voices[label] = ("design", instr.strip())
            else:
                return None, (
                    f"Speaker {label} is used in the script but has no "
                    "reference audio or instruct."
                )

        lang = language if (language and language != "Auto") else None
        seed_val = int(seed) if seed is not None else None

        try:
            gap = np.zeros(
                int(INTER_TURN_GAP_S * sampling_rate), dtype=np.float32
            )
            segments = []
            for label, text in turns:
                mode, val = voices[label]
                kw: Dict[str, Any] = dict(text=text, language=lang, seed=seed_val)
                if mode == "clone":
                    kw["voice_clone_prompt"] = val
                else:
                    kw["instruct"] = val
                audio = model.generate(**kw)[0]
                if segments:
                    segments.append(gap)
                segments.append(audio.astype(np.float32))
            conversation = np.concatenate(segments)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (conversation * 32767).astype(np.int16)
        return (sampling_rate, waveform), f"Done. {len(turns)} turns."

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
            sd = gr.Slider(
                0.0,
                1.0,
                value=0.3,
                step=0.05,
                label="Chunk Silence (seconds)",
                info="Silence + cross-fade inserted between chunks for long "
                "text. 0 = hard concat (may click at seams).",
            )
            se = gr.Number(
                value=None,
                precision=0,
                label="Seed",
                info="Set for reproducible output. Leave empty for random.",
            )
        return ns, gs, dn, sp, du, pp, po, sd, se

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        with gr.Accordion("Saved Voices", open=False):
                            vc_saved = gr.Dropdown(
                                label="Use a saved voice (skips reference above)",
                                choices=_saved_choices(),
                                value=_NO_SAVED_VOICE,
                                info="Pick a previously saved voice to clone "
                                "without re-uploading or re-encoding.",
                            )
                            with gr.Row():
                                vc_save_name = gr.Textbox(
                                    label="Save current reference as",
                                    placeholder="my_voice",
                                    scale=3,
                                )
                                vc_save_btn = gr.Button("Save", scale=1)
                            vc_refresh_btn = gr.Button("Refresh list")
                            vc_save_status = gr.Textbox(
                                label="Saved Voice Status",
                                lines=1,
                                interactive=False,
                            )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                            vc_sd,
                            vc_se,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text,
                    lang,
                    ref_aud,
                    ref_text,
                    instruct,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                    sd,
                    se,
                    saved,
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        sd,
                        se,
                        mode="clone",
                        ref_text=ref_text or None,
                        saved_voice=saved,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                        vc_sd,
                        vc_se,
                        vc_saved,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                vc_save_btn.click(
                    _save_voice,
                    inputs=[vc_ref_audio, vc_ref_text, vc_save_name],
                    outputs=[vc_save_status, vc_saved],
                )
                vc_refresh_btn.click(
                    lambda: gr.update(choices=_saved_choices()),
                    outputs=[vc_saved],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                            vd_sd,
                            vd_se,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)
                        with gr.Accordion(
                            "Save this voice for reuse", open=False
                        ):
                            vd_save_name = gr.Textbox(
                                label="Save the generated voice as",
                                placeholder="my_designed_voice",
                                info="Captures the voice you just generated so "
                                "it can be reused from the Voice Clone tab.",
                            )
                            vd_save_btn = gr.Button("Save voice")
                            vd_save_status = gr.Textbox(
                                label="Saved Voice Status",
                                lines=1,
                                interactive=False,
                            )

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(
                    text, lang, ns, gs, dn, sp, du, pp, po, sd, se, *groups
                ):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        sd,
                        se,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                        vd_sd,
                        vd_se,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

                # Capture the generated design voice; also refresh the Voice
                # Clone tab's saved-voice dropdown so it is reusable there.
                vd_save_btn.click(
                    _save_design_voice,
                    inputs=[vd_audio, vd_text, vd_save_name],
                    outputs=[vd_save_status, vc_saved],
                )

            # ==============================================================
            # Dialogue (multi-speaker)
            # ==============================================================
            with gr.TabItem("Dialogue"):
                gr.Markdown(
                    "Synthesize a two-speaker conversation into one audio. "
                    "Configure **Speaker A** and **Speaker B** below, then "
                    "write the script with one turn per line, prefixed by "
                    "`A:` or `B:`. Each speaker uses reference audio if "
                    "provided, otherwise the instruct text."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Speaker A")
                            dlg_a_audio = gr.Audio(
                                label="A — Reference Audio (optional)",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            dlg_a_ref_text = gr.Textbox(
                                label="A — Reference Text (optional)", lines=1
                            )
                            dlg_a_instruct = gr.Textbox(
                                label="A — Instruct (used if no reference audio)",
                                lines=1,
                                placeholder="e.g. male, middle-aged",
                            )
                        with gr.Group():
                            gr.Markdown("### Speaker B")
                            dlg_b_audio = gr.Audio(
                                label="B — Reference Audio (optional)",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            dlg_b_ref_text = gr.Textbox(
                                label="B — Reference Text (optional)", lines=1
                            )
                            dlg_b_instruct = gr.Textbox(
                                label="B — Instruct (used if no reference audio)",
                                lines=1,
                                placeholder="e.g. female, young adult",
                            )
                        dlg_script = gr.Textbox(
                            label="Dialogue Script",
                            lines=8,
                            placeholder="A: Hi there, how are you?\n"
                            "B: I'm great, thanks for asking!\n"
                            "A: Glad to hear it.",
                        )
                        dlg_lang = _lang_dropdown()
                        dlg_seed = gr.Number(
                            value=None,
                            precision=0,
                            label="Seed",
                            info="Set for reproducible output. Empty = random.",
                        )
                        dlg_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        dlg_audio = gr.Audio(
                            label="Conversation / 对话结果",
                            type="numpy",
                        )
                        dlg_status = gr.Textbox(label="Status / 状态", lines=2)

                dlg_btn.click(
                    _run_dialogue,
                    inputs=[
                        dlg_a_audio,
                        dlg_a_ref_text,
                        dlg_a_instruct,
                        dlg_b_audio,
                        dlg_b_ref_text,
                        dlg_b_instruct,
                        dlg_script,
                        dlg_lang,
                        dlg_seed,
                    ],
                    outputs=[dlg_audio, dlg_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint, prompts_dir=args.prompts_dir)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
