"""
Apollo — one-click AI cover pipeline for Google Colab (free tier friendly)
===========================================================================
Runs on top of Applio (https://github.com/IAHispano/Applio). Run Applio's
Setup cell first so that /content/Applio exists, then run this script.

What it does, end to end, with no manual steps:
    download song -> UVR vocal separation (vocals + instrumental)
    -> RVC voice conversion on the vocals ONLY (your trained model)
    -> loudness-normalize the converted vocals
    -> mix back with the ORIGINAL instrumental -> final wav + mp3

Your voice model can be auto-downloaded from a HuggingFace repo (public or
private) so a fresh Colab session needs no manual file wrangling:
    - put a *Read* token in Colab Secrets under the name HF_TOKEN
      (huggingface.co/settings/tokens), or pass the repo as public.
    - the notebook cell must first copy the secret into the environment,
      because `userdata` is only readable by the cell process itself:

        from google.colab import userdata
        import os
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        !wget -q "<raw-url-of-this-file>" -O /content/apollo_pipeline.py
        !python /content/apollo_pipeline.py \
            --song <direct-audio-url> \
            --model-name MyVoice --model-repo you/MyVoice

Only test the model download (no Applio needed):
        !python /content/apollo_pipeline.py --fetch-model \
            --model-name MyVoice --model-repo you/MyVoice

Options:
    --song URL        direct link to the song audio (mp3/wav) [required]
    --model-name      model base name; expects <name>.pth + <name>.index
                      in Applio's logs dir or in the HF repo
    --model-repo      HuggingFace repo id hosting the model files
    --pth-name        file name of the .pth inside the repo (default
                      derived from --model-name)
    --pitch N         transpose semitones, e.g. 2 or -2 (default 0)
    --index-rate F    how strongly the index shapes the timbre (default 0.75)
    --protect F       consonant/breath protection (default 0.33)
    --out NAME        output file base name (default apollo_cover)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request

# ================= defaults (override via CLI / env) =================
APPLIO_DIR = "/content/Applio"
WORK_DIR = "/content/apollo_work"

F0_METHOD = "rmvpe"               # match what the model was trained with
EMBEDDER = "chinese-hubert-base"  # content embedder used in training
# ======================================================================


def log(msg):
    print(f"[apollo] {msg}", flush=True)


def run(cmd):
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def hf_token():
    """HF token: Colab Secrets (HF_TOKEN) -> environment variable."""
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_from_hf(base, cfg):
    """Download <name>.pth + <name>.index from the HF repo into Applio logs."""
    if not cfg.model_repo:
        raise SystemExit(
            "[apollo] Model not found locally and no --model-repo given.\n"
            "  Either upload the model files into the Colab session at\n"
            f"  {base}/  or pass --model-repo <user>/<repo>."
        )
    tok = hf_token()
    from huggingface_hub import hf_hub_download
    os.makedirs(base, exist_ok=True)
    log(f"Downloading model from HuggingFace repo {cfg.model_repo} ...")
    p = hf_hub_download(cfg.model_repo, cfg.pth_name, token=tok)
    i = hf_hub_download(cfg.model_repo, cfg.index_name, token=tok)
    pth = os.path.join(base, f"{cfg.model_name}.pth")
    idx = os.path.join(base, f"{cfg.model_name}.index")
    shutil.copy(p, pth)
    shutil.copy(i, idx)
    log(f"Model ready: {pth}")
    return pth, idx


def model_paths(cfg):
    """Local candidates first, then HuggingFace auto-download."""
    base = f"{APPLIO_DIR}/logs/{cfg.model_name}"
    cand_pth = [f"{base}/{cfg.model_name}.pth", f"/content/{cfg.model_name}.pth",
                f"{base}/{cfg.pth_name}"]
    cand_idx = [f"{base}/{cfg.model_name}.index", f"/content/{cfg.model_name}.index",
                f"{base}/{cfg.index_name}"]
    pth = next((p for p in cand_pth if os.path.exists(p)), None)
    idx = next((p for p in cand_idx if os.path.exists(p)), None)
    if pth and idx:
        log(f"Found model: {pth}")
        return pth, idx
    return download_from_hf(base, cfg)


def main():
    ap = argparse.ArgumentParser(description="Apollo one-click RVC cover pipeline")
    ap.add_argument("--song", help="direct URL of the song audio (mp3/wav)")
    ap.add_argument("--out", default="apollo_cover", help="output base name")
    ap.add_argument("--pitch", type=int, default=0, help="transpose, e.g. 2 / -2")
    ap.add_argument("--model-name", default=os.environ.get("APOLLO_MODEL_NAME", "MyVoice"))
    ap.add_argument("--model-repo", default=os.environ.get("APOLLO_HF_REPO", ""))
    ap.add_argument("--pth-name", default="")
    ap.add_argument("--index-name", default="")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--fetch-model", action="store_true",
                    help="only download the model and exit (wiring test)")
    cfg = ap.parse_args()
    if not cfg.pth_name:
        cfg.pth_name = f"{cfg.model_name}.pth"
    if not cfg.index_name:
        cfg.index_name = f"{cfg.model_name}.index"

    os.makedirs(WORK_DIR, exist_ok=True)

    # ---- wiring test: model download only (Applio not required) ----
    if cfg.fetch_model:
        pth, idx = model_paths(cfg)
        log(f"Model download OK: {pth} ({os.path.getsize(pth) / 1e6:.1f} MB) / "
            f"{idx} ({os.path.getsize(idx) / 1e6:.1f} MB)")
        return

    if not cfg.song:
        raise SystemExit("[apollo] --song <direct-audio-url> is required")

    # ---- 0. environment check ----
    if not os.path.isdir(APPLIO_DIR):
        raise SystemExit("[apollo] /content/Applio not found — run Applio's Setup cell first")
    model_pth, index_path = model_paths(cfg)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        log("GPU: " + (smi.stdout.strip() or "not detected (will be slow)"))
    except Exception:
        log("GPU: not detected (will be slow)")

    # ---- 1. dependencies (Applio inference + UVR separator) ----
    log("Installing dependencies (audio-separator, torchcrepe, faiss-cpu)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "audio-separator[gpu]", "torchcrepe", "faiss-cpu", "huggingface_hub"],
        check=True)

    # ---- 2. download the song ----
    ext = os.path.splitext(cfg.song.split("?")[0])[1] or ".wav"
    song_path = os.path.join(WORK_DIR, "song" + ext)
    if not os.path.exists(song_path):
        log("Downloading song...")
        urllib.request.urlretrieve(cfg.song, song_path)
    log(f"Song: {song_path} ({os.path.getsize(song_path) / 1e6:.1f} MB)")

    # ---- 3. UVR separation: vocals + instrumental ----
    log("Separating vocals / instrumental (a few minutes)...")
    from audio_separator.separator import Separator

    sep = Separator(output_dir=WORK_DIR, output_format="WAV",
                    model_file_dir=os.path.join(WORK_DIR, "uvr_models"))
    sep.load_model("UVR-MDX-NET-Voc_FT.onnx")
    sep.separate(song_path)
    vocals = glob.glob(os.path.join(WORK_DIR, "*_(Vocals)_*.wav"))
    insts = glob.glob(os.path.join(WORK_DIR, "*_(Instrumental)_*.wav"))
    if not vocals or not insts:
        raise SystemExit("[apollo] UVR separation failed, output dir: " + WORK_DIR)
    vocal_path, inst_path = vocals[0], insts[0]
    log(f"Vocals: {os.path.basename(vocal_path)}")
    log(f"Instrumental: {os.path.basename(inst_path)}")

    # ---- 4. RVC conversion on the vocals only ----
    log("Converting vocals (a ~5 min song takes ~6-15 min on a free T4)...")
    sys.path.insert(0, APPLIO_DIR)
    os.chdir(APPLIO_DIR)  # Applio uses relative paths internally
    from rvc.infer.infer import VoiceConverter

    vc = VoiceConverter()
    conv_wav = os.path.join(WORK_DIR, "vocal_converted.wav")
    vc.convert_audio(
        audio_input_path=vocal_path,
        audio_output_path=conv_wav,
        model_path=model_pth,
        index_path=index_path,
        pitch=cfg.pitch,
        f0_method=F0_METHOD,
        index_rate=cfg.index_rate,
        protect=cfg.protect,
        embedder_model=EMBEDDER,
        export_format="WAV",
    )

    # ---- 5. loudness-normalize vocals, mix with original instrumental ----
    log("Normalizing loudness and mixing...")
    conv_norm = os.path.join(WORK_DIR, "vocal_converted_norm.wav")
    run(["ffmpeg", "-y", "-i", conv_wav, "-filter:a",
         "loudnorm=I=-16:TP=-1.5:LRA=11", conv_norm])
    final_wav = f"/content/{cfg.out}.wav"
    final_mp3 = f"/content/{cfg.out}.mp3"
    run(["ffmpeg", "-y", "-i", conv_norm, "-i", inst_path,
         "-filter_complex",
         "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0",
         "-c:a", "pcm_s16le", final_wav])
    run(["ffmpeg", "-y", "-i", final_wav, "-codec:a", "libmp3lame", "-q:a", "2", final_mp3])

    # ---- 6. done ----
    for f in (final_wav, final_mp3):
        log(f"Output: {f} ({os.path.getsize(f) / 1e6:.1f} MB)")
    if os.path.isdir("/content/drive/MyDrive"):
        shutil.copy(final_mp3, "/content/drive/MyDrive/")
        log("Also copied to Google Drive")
    log("Done! Download the result from the Colab file panel.")


if __name__ == "__main__":
    main()
