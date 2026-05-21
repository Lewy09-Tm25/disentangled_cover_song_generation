import os
import subprocess
import logging
import itertools
import random
import pandas as pd

# Configure logging to display real-time progress and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_youtube_audio(yt_id, work_id, perf_id, output_dir):
    """
    Attempts to download a single YouTube video as a wav file.
    Returns True if successful or if the file already exists, False otherwise.
    """
    file_name = f"{work_id}_{perf_id}"
    output_template = os.path.join(output_dir, f"{file_name}.%(ext)s")
    expected_output = os.path.join(output_dir, f"{file_name}.wav")
    
    # Checkpoint check for individual files
    if os.path.exists(expected_output):
        logging.info(f"File already exists on disk: {expected_output}")
        return True
        
    yt_link = f"https://www.youtube.com/watch?v={yt_id}"
    command = ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_template, yt_link]
    
    try:
        # Run yt-dlp quietly, only capturing output to determine success
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logging.info(f"Successfully downloaded: {yt_link}")
            return True
        else:
            logging.warning(f"Failed to download: {yt_link} | Error Snippet: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"System Exception during download of {yt_link}: {str(e)}")
        return False

def main():
    # Setup directories and targets
    output_dir = "audio_downloads"
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = "extra_datasets/shs-100k/train.csv"
    validate_file = "extra_datasets/shs-100k/validate.csv"
    output_csv = "downloaded_pairs.csv"
    
    # Expected columns as the CSVs do not contain headers
    col_names = ["performance_id", "work_id", "performance_title", "performing_artist", "youtube_id"]
    
    # Ingest and combine data
    try:
        df_train = pd.read_csv(train_file, header=None, names=col_names)
        df_val = pd.read_csv(validate_file, header=None, names=col_names)
        df_all = pd.concat([df_train, df_val], ignore_index=True)
    except FileNotFoundError as e:
        logging.error(f"Input file missing. Ensure train.csv and validate.csv are in the directory. Details: {e}")
        return
        
    # Checkpoint Configuration
    processed_works = set()
    out_columns = [
        "work_id", 
        "perf_id_A", "title_A", "artist_A", "yt_id_A",
        "perf_id_B", "title_B", "artist_B", "yt_id_B"
    ]
    
    # Load checkpoint if it exists, otherwise create the file with headers
    if os.path.exists(output_csv):
        df_checkpoint = pd.read_csv(output_csv)
        processed_works = set(df_checkpoint["work_id"].unique())
        logging.info(f"Resuming from checkpoint. {len(processed_works)} work_ids already fully processed.")
    else:
        pd.DataFrame(columns=out_columns).to_csv(output_csv, index=False)
        logging.info("Starting fresh. Created new checkpoint lookup file.")

    # Group the dataframe by work_id to locate covers
    grouped = df_all.groupby('work_id')
    
    for work_id, group in grouped:
        if work_id in processed_works:
            continue
            
        if len(group) < 2:
            logging.debug(f"Skipping work_id {work_id}: Less than 2 performances available (no cover).")
            continue
            
        logging.info(f"--- Processing work_id: {work_id} ---")
        
        # Extract rows to dictionaries for easy pair generation
        performances = group.to_dict('records')
        
        # Generate all permutations (pairs) and shuffle them to ensure randomness
        pairs = list(itertools.combinations(performances, 2))
        random.shuffle(pairs)
        
        pair_success = False
        failed_yts = set() # Track dead links specific to this work_id so we don't retry them
        
        for perf_A, perf_B in pairs:
            yt_A = perf_A['youtube_id']
            yt_B = perf_B['youtube_id']
            
            # Instantly bypass this combination if we already know one of the links is dead
            if yt_A in failed_yts or yt_B in failed_yts:
                continue
                
            logging.info(f"Attempting Pair: {perf_A['performance_id']} & {perf_B['performance_id']}")
            
            # Attempt Song A
            success_A = download_youtube_audio(yt_A, work_id, perf_A['performance_id'], output_dir)
            if not success_A:
                failed_yts.add(yt_A)
                continue
                
            # Attempt Song B
            success_B = download_youtube_audio(yt_B, work_id, perf_B['performance_id'], output_dir)
            if not success_B:
                failed_yts.add(yt_B)
                continue
                
            # If code reaches here, both files downloaded successfully
            pair_success = True
            
            # Immediately append to the CSV to checkpoint the progress
            new_row = pd.DataFrame([{
                "work_id": work_id,
                "perf_id_A": perf_A["performance_id"],
                "title_A": perf_A["performance_title"],
                "artist_A": perf_A["performing_artist"],
                "yt_id_A": yt_A,
                "perf_id_B": perf_B["performance_id"],
                "title_B": perf_B["performance_title"],
                "artist_B": perf_B["performing_artist"],
                "yt_id_B": yt_B
            }])
            new_row.to_csv(output_csv, mode='a', header=False, index=False)
            
            logging.info(f"SUCCESS: Locked and checkpointed pair for work_id {work_id}")
            break # Break the permutations loop, move to the next work_id
            
        if not pair_success:
            logging.warning(f"EXHAUSTED: All combinations for work_id {work_id} failed. Skipping entirely.")

if __name__ == "__main__":
    main()