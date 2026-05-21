import os
import logging
import pandas as pd
import librosa
import soundfile as sf
import warnings

# Suppress librosa warnings for cleaner console output
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global Audio Parameters
TARGET_SR = 22050

def main():
    # Input and Output Directories
    input_dir = "chorus_audio"
    output_dir = "augmented_chorus_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpointing Files
    input_csv = "dtw_aligned_pairs.csv"
    output_csv = "augmented_pairs.csv"
    
    try:
        df_aligned = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Cannot find {input_csv}. Ensure DTW extraction is complete.")
        return

    # Setup Checkpoint Dataframe
    processed_works = set()
    out_cols = [
        "work_id", "shift_amount", 
        "anchor_perf_id", "target_perf_id", 
        "anchor_audio_file", "target_audio_file"
    ]
    
    if os.path.exists(output_csv):
        df_check = pd.read_csv(output_csv)
        processed_works = set(df_check["work_id"].unique())
        logging.info(f"Resuming: {len(processed_works)} Work IDs already fully augmented.")
    else:
        pd.DataFrame(columns=out_cols).to_csv(output_csv, index=False)

    # Main Augmentation Loop
    for idx, row in df_aligned.iterrows():
        work_id = row['work_id']
        if work_id in processed_works:
            continue
            
        logging.info(f"\n--- Augmenting Work ID: {work_id} ---")
        
        anchor_id = row['anchor_perf_id']
        target_id = row['target_perf_id']
        
        anchor_path = os.path.join(input_dir, row['anchor_audio_file'])
        target_path = os.path.join(input_dir, row['target_audio_file'])
        
        if not os.path.exists(anchor_path) or not os.path.exists(target_path):
            logging.warning(f"Source audio missing for Work ID {work_id}. Skipping.")
            continue
            
        try:
            # Load the pristine 30-second choruses into memory
            y_anchor, _ = librosa.load(anchor_path, sr=TARGET_SR, mono=True)
            y_target, _ = librosa.load(target_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            logging.error(f"Failed to load audio for Work ID {work_id}: {e}")
            continue

        new_rows = []

        # ==========================================
        # 1. BASE PAIR (0 Shift)
        # ==========================================
        # We rewrite the base files to the new directory to keep the final 
        # dataset unified in one place for the tensor extraction phase.
        base_anchor_out = f"{work_id}_{anchor_id}_Anchor_shift0.wav"
        base_target_out = f"{work_id}_{target_id}_Target_shift0.wav"
        
        sf.write(os.path.join(output_dir, base_anchor_out), y_anchor, TARGET_SR)
        sf.write(os.path.join(output_dir, base_target_out), y_target, TARGET_SR)
        
        new_rows.append({
            "work_id": work_id, "shift_amount": 0,
            "anchor_perf_id": anchor_id, "target_perf_id": target_id,
            "anchor_audio_file": base_anchor_out, "target_audio_file": base_target_out
        })

        # ==========================================
        # 2. SHIFT +1 SEMITONE
        # ==========================================
        logging.info("Applying +1 semitone shift (Compute heavy)...")
        try:
            y_anchor_plus1 = librosa.effects.pitch_shift(y_anchor, sr=TARGET_SR, n_steps=1)
            y_target_plus1 = librosa.effects.pitch_shift(y_target, sr=TARGET_SR, n_steps=1)
            
            plus1_anchor_out = f"{work_id}_{anchor_id}_Anchor_shift+1.wav"
            plus1_target_out = f"{work_id}_{target_id}_Target_shift+1.wav"
            
            sf.write(os.path.join(output_dir, plus1_anchor_out), y_anchor_plus1, TARGET_SR)
            sf.write(os.path.join(output_dir, plus1_target_out), y_target_plus1, TARGET_SR)
            
            new_rows.append({
                "work_id": work_id, "shift_amount": 1,
                "anchor_perf_id": anchor_id, "target_perf_id": target_id,
                "anchor_audio_file": plus1_anchor_out, "target_audio_file": plus1_target_out
            })
        except Exception as e:
            logging.error(f"+1 Shift failed for {work_id}: {e}")

        # ==========================================
        # 3. SHIFT -1 SEMITONE
        # ==========================================
        logging.info("Applying -1 semitone shift (Compute heavy)...")
        try:
            y_anchor_minus1 = librosa.effects.pitch_shift(y_anchor, sr=TARGET_SR, n_steps=-1)
            y_target_minus1 = librosa.effects.pitch_shift(y_target, sr=TARGET_SR, n_steps=-1)
            
            minus1_anchor_out = f"{work_id}_{anchor_id}_Anchor_shift-1.wav"
            minus1_target_out = f"{work_id}_{target_id}_Target_shift-1.wav"
            
            sf.write(os.path.join(output_dir, minus1_anchor_out), y_anchor_minus1, TARGET_SR)
            sf.write(os.path.join(output_dir, minus1_target_out), y_target_minus1, TARGET_SR)
            
            new_rows.append({
                "work_id": work_id, "shift_amount": -1,
                "anchor_perf_id": anchor_id, "target_perf_id": target_id,
                "anchor_audio_file": minus1_anchor_out, "target_audio_file": minus1_target_out
            })
        except Exception as e:
            logging.error(f"-1 Shift failed for {work_id}: {e}")

        # ==========================================
        # CHECKPOINT
        # ==========================================
        pd.DataFrame(new_rows).to_csv(output_csv, mode='a', header=False, index=False)
        logging.info(f"SUCCESS: Multiplied and checkpointed pairs for Work ID {work_id}")

if __name__ == "__main__":
    main()