import os
import subprocess
import logging
import pandas as pd
import numpy as np
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
HOP_LENGTH = 4096
DURATION_SEC = 30

def download_youtube_audio(yt_id, work_id, perf_id, output_dir):
    """Downloads a fallback audio track."""
    file_name = f"{work_id}_{perf_id}"
    output_template = os.path.join(output_dir, f"{file_name}.%(ext)s")
    expected_output = os.path.join(output_dir, f"{file_name}.wav")
    
    if os.path.exists(expected_output):
        return True, expected_output
        
    yt_link = f"https://www.youtube.com/watch?v={yt_id}"
    command = ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_template, yt_link]
    
    try:
        logging.info(f"Downloading healing candidate: {yt_link}")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return True, expected_output
        else:
            logging.warning(f"Failed to download candidate {yt_link} | Error Snippet: {result.stderr.strip()[-100:]}")
            return False, None
    except Exception as e:
        logging.error(f"Download exception for {yt_link}: {str(e)}")
        return False, None

def find_highest_energy_segment(y, sr, duration_sec):
    """Finds the loudest segment (chorus) using RMS energy."""
    rms = librosa.feature.rms(y=y)[0]
    frames_per_sec = sr / 512 
    window_length = int(duration_sec * frames_per_sec)
    
    if len(rms) < window_length:
        return 0.0, len(y) / sr 
        
    window = np.ones(window_length) / window_length
    smoothed_rms = np.convolve(rms, window, mode='valid')
    max_frame = np.argmax(smoothed_rms)
    
    start_time = librosa.frames_to_time(max_frame, sr=sr, hop_length=512)
    end_time = start_time + duration_sec
    return start_time, end_time

def process_anchor(in_path, out_path):
    """Attempts to isolate the chorus and compute the chromagram. Acts as the base for DTW."""
    try:
        if not os.path.exists(in_path):
            return False, None, None, None
            
        y_anchor, _ = librosa.load(in_path, sr=TARGET_SR, mono=True)
        if len(y_anchor) < TARGET_SR * DURATION_SEC:
            return False, None, None, None

        start_sec, end_sec = find_highest_energy_segment(y_anchor, TARGET_SR, DURATION_SEC)
        
        # Slicing & Saving
        anchor_slice = y_anchor[int(start_sec * TARGET_SR) : int(end_sec * TARGET_SR)]
        sf.write(out_path, anchor_slice, TARGET_SR)
        
        # Heavy Math: Extract Chromagram for the full track to use in DTW
        chroma = librosa.feature.chroma_cqt(y=y_anchor, sr=TARGET_SR, hop_length=HOP_LENGTH)
        
        return True, chroma, start_sec, end_sec
        
    except Exception as e:
        logging.debug(f"Anchor extraction failed for {in_path}: {e}")
        return False, None, None, None

def process_target_candidate(target_path, out_path, chroma_anchor, anchor_start_sec, anchor_end_sec):
    """Runs DTW against the pre-calculated Anchor chromagram."""
    try:
        if not os.path.exists(target_path):
            return False
            
        y_target, _ = librosa.load(target_path, sr=TARGET_SR, mono=True)
        if len(y_target) < TARGET_SR * DURATION_SEC:
            return False

        chroma_target = librosa.feature.chroma_cqt(y=y_target, sr=TARGET_SR, hop_length=HOP_LENGTH)
        
        # DTW Alignment
        D, wp = librosa.sequence.dtw(X=chroma_anchor, Y=chroma_target, metric='cosine')
        wp = wp[::-1, :] 

        # Map timestamps
        anchor_start_frame = librosa.time_to_frames(anchor_start_sec, sr=TARGET_SR, hop_length=HOP_LENGTH)
        anchor_end_frame = librosa.time_to_frames(anchor_end_sec, sr=TARGET_SR, hop_length=HOP_LENGTH)

        target_start_idx = np.searchsorted(wp[:, 0], anchor_start_frame)
        target_end_idx = np.searchsorted(wp[:, 0], anchor_end_frame)
        
        target_start_idx = min(target_start_idx, len(wp) - 1)
        target_end_idx = min(target_end_idx, len(wp) - 1)

        target_start_sec = librosa.frames_to_time(wp[target_start_idx, 1], sr=TARGET_SR, hop_length=HOP_LENGTH)
        target_end_sec = librosa.frames_to_time(wp[target_end_idx, 1], sr=TARGET_SR, hop_length=HOP_LENGTH)
        
        target_slice = y_target[int(target_start_sec * TARGET_SR) : int(target_end_sec * TARGET_SR)]

        # Check safety boundary
        if len(target_slice) < TARGET_SR:
            return False

        sf.write(out_path, target_slice, TARGET_SR)
        return True
        
    except Exception as e:
        return False

def main():
    audio_dir = "audio_downloads"
    chorus_dir = "chorus_audio"
    os.makedirs(chorus_dir, exist_ok=True)
    
    input_pairs_file = "downloaded_pairs.csv"
    output_checkpoint = "dtw_aligned_pairs.csv"
    col_names = ["performance_id", "work_id", "performance_title", "performing_artist", "youtube_id"]
    
    try:
        df_pairs = pd.read_csv(input_pairs_file)
        df_train = pd.read_csv("extra_datasets/shs-100k/train.csv", header=None, names=col_names)
        df_val = pd.read_csv("extra_datasets/shs-100k/validate.csv", header=None, names=col_names)
        df_corpus = pd.concat([df_train, df_val], ignore_index=True)
    except FileNotFoundError as e:
        logging.error(f"Required CSV missing: {e}")
        return

    # Checkpoint Handling
    processed_works = set()
    out_cols = ["work_id", "anchor_perf_id", "target_perf_id", "anchor_audio_file", "target_audio_file"]
    
    if os.path.exists(output_checkpoint):
        df_check = pd.read_csv(output_checkpoint)
        processed_works = set(df_check["work_id"].unique())
        logging.info(f"Resuming: {len(processed_works)} work_ids already aligned.")
    else:
        pd.DataFrame(columns=out_cols).to_csv(output_checkpoint, index=False)

    # Main Processing Loop
    for idx, row in df_pairs.iterrows():
        work_id = row['work_id']
        if work_id in processed_works:
            continue
            
        logging.info(f"\n--- Aligning Work ID: {work_id} ---")
        
        perf_A = row['perf_id_A']
        perf_B = row['perf_id_B']
        
        path_A = os.path.join(audio_dir, f"{work_id}_{perf_A}.wav")
        path_B = os.path.join(audio_dir, f"{work_id}_{perf_B}.wav")
        
        anchor_perf = None
        target_to_try_first = None
        full_length_anchor_path = None
        
        # 1. ATTEMPT SONG A AS ANCHOR
        out_anchor_path = os.path.join(chorus_dir, f"{work_id}_{perf_A}_Anchor_Chorus.wav")
        success_anchor, chroma_anchor, start_anchor, end_anchor = process_anchor(path_A, out_anchor_path)
        
        if success_anchor:
            logging.info(f"Song A ({perf_A}) successfully set as Anchor.")
            anchor_perf = perf_A
            target_to_try_first = perf_B
            full_length_anchor_path = path_A
            # Cleanup the failed B path if it exists, since we won't use it as an anchor
            if os.path.exists(path_B):
                 os.remove(path_B)
        else:
            logging.warning(f"Song A ({perf_A}) failed. Deleting and attempting Song B ({perf_B}) as Anchor...")
            if os.path.exists(path_A):
                os.remove(path_A) # CLEANUP: Delete failed Song A
                
            # 2. ATTEMPT SONG B AS ANCHOR (Since A failed)
            out_anchor_path = os.path.join(chorus_dir, f"{work_id}_{perf_B}_Anchor_Chorus.wav")
            success_anchor, chroma_anchor, start_anchor, end_anchor = process_anchor(path_B, out_anchor_path)
            
            if success_anchor:
                logging.info(f"Song B ({perf_B}) successfully set as Anchor.")
                anchor_perf = perf_B
                target_to_try_first = None 
                full_length_anchor_path = path_B
            else:
                logging.error(f"FATAL: Both {perf_A} and {perf_B} failed to anchor. Skipping Work ID {work_id}.")
                if os.path.exists(path_B):
                    os.remove(path_B) # CLEANUP: Delete failed Song B
                continue 

        # 3. BUILD CANDIDATE TARGET LIST
        corpus_candidates = df_corpus[(df_corpus['work_id'] == work_id) & 
                                      (df_corpus['performance_id'] != perf_A) & 
                                      (df_corpus['performance_id'] != perf_B)]
        
        candidates_list = corpus_candidates.to_dict('records')
        
        if target_to_try_first:
            first_cand = {'performance_id': perf_B, 'youtube_id': row['yt_id_B']}
            candidates_list.insert(0, first_cand)

        # --- COMPUTE OPTIMIZATION: CAP ATTEMPTS ---
        MAX_TARGET_ATTEMPTS = 6
        candidates_list = candidates_list[:MAX_TARGET_ATTEMPTS]
        logging.info(f"Capping target attempts to {len(candidates_list)} to save compute time.")

        pair_formed = False
        
        # 4. THE HEALING LOOP 
        for cand in candidates_list:
            cand_perf = cand['performance_id']
            cand_yt = cand['youtube_id']
            
            logging.info(f"Testing Candidate Target: {cand_perf}")
            out_target_path = os.path.join(chorus_dir, f"{work_id}_{cand_perf}_Target_Chorus.wav")
            
            dl_success, in_target_path = download_youtube_audio(cand_yt, work_id, cand_perf, audio_dir)
            if not dl_success:
                continue
                
            success_target = process_target_candidate(in_target_path, out_target_path, chroma_anchor, start_anchor, end_anchor)
            
            if success_target:
                logging.info(f"SUCCESS: DTW Aligned Anchor {anchor_perf} with Target {cand_perf}")
                
                new_row = pd.DataFrame([{
                    "work_id": work_id,
                    "anchor_perf_id": anchor_perf,
                    "target_perf_id": cand_perf,
                    "anchor_audio_file": os.path.basename(out_anchor_path),
                    "target_audio_file": os.path.basename(out_target_path)
                }])
                new_row.to_csv(output_checkpoint, mode='a', header=False, index=False)
                
                pair_formed = True
                
                # CLEANUP: Delete the full-length target now that we have the 30s chorus
                if os.path.exists(in_target_path):
                    os.remove(in_target_path)
                break 
            else:
                logging.warning(f"DTW Failed for Target {cand_perf}. Deleting file and trying next...")
                # CLEANUP: Instantly delete the failed candidate audio
                if os.path.exists(in_target_path):
                    os.remove(in_target_path)

        # 5. POST-LOOP CLEANUP & EXHAUSTION PROTOCOL
        if pair_formed:
            # We successfully formed a pair. Delete the full-length Anchor track to save space.
            if full_length_anchor_path and os.path.exists(full_length_anchor_path):
                os.remove(full_length_anchor_path)
        else:
            logging.error(f"EXHAUSTED: All {len(candidates_list)} targets failed for Anchor {anchor_perf}. Deleting orphan Anchor.")
            if os.path.exists(out_anchor_path):
                os.remove(out_anchor_path) # Delete the 30s anchor slice
            if full_length_anchor_path and os.path.exists(full_length_anchor_path):
                os.remove(full_length_anchor_path) # Delete the full length anchor

if __name__ == "__main__":
    main()