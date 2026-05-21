import os
import shutil
import logging
import pandas as pd
import numpy as np
import torch
import torchaudio
import librosa
import warnings
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import Wav2Vec2FeatureExtractor, AutoModel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# OFFLINE CONFIGURATION
# ---------------------------------------------------------
# Safely fallback to CPU since you are running this locally without a heavy GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running offline extraction on: {DEVICE.upper()}")

INPUT_DIR = "augmented_chorus_audio" 
TENSOR_DIR = "tensor_dataset"        
os.makedirs(TENSOR_DIR, exist_ok=True)

# ---------------------------------------------------------
# MODEL INITIALIZATION (Frozen Extractors)
# ---------------------------------------------------------
logging.info("Loading pre-trained models into local memory...")

# 1. MERT (Style Extractor)
MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
mert_model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True).to(DEVICE)
mert_model.eval()

# 2. EnCodec (Target Tokenizer)
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0) 
encodec_model.to(DEVICE)
encodec_model.eval()

logging.info("Models loaded successfully. Starting extraction...")

# ---------------------------------------------------------
# EXTRACTION FUNCTIONS
# ---------------------------------------------------------

def extract_content_chromagram(audio_path, save_path):
    """Extracts 12-bin Chromagram using librosa (Runs on CPU)."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096)
    np.save(save_path, chroma)

@torch.no_grad() 
def extract_style_mert(audio_path, save_path):
    """Extracts style vector from MERT Layer 3 (Bypassing torchaudio)."""
    # FIXED: MERT specifically requires 24kHz, not 16kHz
    TARGET_SR = 24000 
    
    # librosa automatically resamples and converts to mono safely on Windows
    wav_np, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    
    inputs = mert_processor(wav_np, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    outputs = mert_model(**inputs, output_hidden_states=True)
    early_layers = torch.stack(outputs.hidden_states[1:4]) 
    
    # Compress into a 1D vector
    style_vector = early_layers.mean(dim=(0, 1, 2)).cpu().numpy()
    np.save(save_path, style_vector)

@torch.no_grad()
def extract_target_tokens(audio_path, save_path):
    """Compresses audio into EnCodec integer tokens (Bypassing torchaudio)."""
    # EnCodec requires exactly 24kHz
    wav_np, sr = librosa.load(audio_path, sr=24000, mono=True)
    
    # Convert numpy array to PyTorch tensor
    # EnCodec expects shape: [Batch, Channels, Time] -> [1, 1, Time]
    wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    encoded_frames = encodec_model.encode(wav_tensor)
    tokens = encoded_frames[0][0] 
    
    torch.save(tokens.squeeze(0).cpu(), save_path)

# ---------------------------------------------------------
# MAIN OFFLINE BATCH PROCESSING
# ---------------------------------------------------------

def main():
    input_csv = "augmented_pairs.csv"
    checkpoint_csv = "tensor_extraction_checkpoint.csv"
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Cannot find {input_csv}.")
        return

    processed_files = set()
    out_cols = ["work_id", "shift_amount", "content_npy", "style_npy", "target_pt"]
    
    if os.path.exists(checkpoint_csv):
        df_check = pd.read_csv(checkpoint_csv)
        processed_files = set(df_check["content_npy"].unique())
        logging.info(f"Resuming: {len(processed_files)} pairs already extracted.")
    else:
        pd.DataFrame(columns=out_cols).to_csv(checkpoint_csv, index=False)

    for idx, row in df.iterrows():
        work_id = row['work_id']
        shift = row['shift_amount']
        anchor_audio = row['anchor_audio_file']
        target_audio = row['target_audio_file']
        
        base_name = f"{work_id}_shift{shift}"
        content_out = f"content_{base_name}.npy"
        style_out = f"style_{base_name}.npy"
        target_out = f"target_{base_name}.pt"
        
        if content_out in processed_files:
            continue

        logging.info(f"Extracting Tensors for Work ID: {work_id} (Shift {shift})")
        
        anchor_path = os.path.join(INPUT_DIR, anchor_audio)
        target_path = os.path.join(INPUT_DIR, target_audio)
        
        try:
            extract_content_chromagram(anchor_path, os.path.join(TENSOR_DIR, content_out))
            extract_style_mert(target_path, os.path.join(TENSOR_DIR, style_out))
            extract_target_tokens(target_path, os.path.join(TENSOR_DIR, target_out))
        except Exception as e:
            logging.error(f"Extraction failed for {base_name}. Reason: {str(e)}")
            continue

        # Checkpoint
        new_row = pd.DataFrame([{
            "work_id": work_id, "shift_amount": shift,
            "content_npy": content_out, "style_npy": style_out, "target_pt": target_out
        }])
        new_row.to_csv(checkpoint_csv, mode='a', header=False, index=False)

    # ---------------------------------------------------------
    # FINAL PACKAGING FOR COLAB
    # ---------------------------------------------------------
    logging.info("Extraction complete. Zipping dataset for Google Drive upload...")
    shutil.make_archive("FINAL_TENSOR_DATASET", 'zip', TENSOR_DIR)
    logging.info("Done! You can now upload 'FINAL_TENSOR_DATASET.zip' and your checkpoint CSV to Google Drive.")

if __name__ == "__main__":
    main()