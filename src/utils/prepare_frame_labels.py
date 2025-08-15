import pretty_midi
import numpy as np
import pandas as pd
import argparse


def extract_frame_labels(midi_path, frames_per_second=100, J=5):
    """
    Extract quantized frame representations from MIDI file.
    
    Args:
        midi_path (str): Path to MIDI file
        frames_per_second (int): Frame rate for quantization (default: 100)
        J (int): Half-width in frames for onset/offset regression spikes (default: 5)
    
    Returns:
        dict: Dictionary containing 'onset', 'offset', and 'frames' arrays
              - onset/offset: Regression targets with triangular spikes (shape: num_frames x 88)
              - frames: Binary activation labels (shape: num_frames x 88)
              Each array covers 88 piano keys (MIDI 21-108: A0 to C8)
    """
    # Piano key range: MIDI 21 (A0) to MIDI 108 (C8) = 88 keys
    MIN_MIDI = 21
    MAX_MIDI = 108
    NUM_PIANO_KEYS = 88
    
    # Frame length in seconds
    frame_len = 1.0 / frames_per_second
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Find the first instrument (should be "Trans A")
    if not midi_data.instruments:
        raise ValueError("No instruments found in MIDI file")
    
    instrument = midi_data.instruments[0]
    print(f"Using instrument: {instrument.name}")
    
    # Get total duration and calculate number of frames
    total_duration = midi_data.get_end_time()
    num_frames = int(np.ceil(total_duration * frames_per_second))
    
    # Initialize arrays for onset, offset, and frames (num_frames x 88)
    onset_frames = np.zeros((num_frames, NUM_PIANO_KEYS), dtype=np.float32)
    offset_frames = np.zeros((num_frames, NUM_PIANO_KEYS), dtype=np.float32)
    note_frames = np.zeros((num_frames, NUM_PIANO_KEYS), dtype=np.float32)
    
    # Process each note
    for note in instrument.notes:
        # Skip notes outside piano range
        if note.pitch < MIN_MIDI or note.pitch > MAX_MIDI:
            continue
            
        # Convert MIDI pitch to piano key index (0-87)
        piano_key_idx = note.pitch - MIN_MIDI
        
        # Exact onset and offset times in seconds
        onset_time = note.start
        offset_time = note.end
        
        # Convert to frame indices for binary frames labels
        onset_frame = int(onset_time * frames_per_second)
        offset_frame = int(offset_time * frames_per_second)
        
        # Set binary frames between onset and offset
        onset_frame_bounded = max(0, min(onset_frame, num_frames - 1))
        offset_frame_bounded = max(0, min(offset_frame, num_frames - 1))
        for frame_idx in range(onset_frame_bounded, min(offset_frame_bounded, num_frames)):
            note_frames[frame_idx, piano_key_idx] = 1.0
        
        # Create regression targets for onset
        for frame_idx in range(max(0, onset_frame - J), min(num_frames, onset_frame + J + 1)):
            # Calculate frame center time
            frame_center_time = (frame_idx + 0.5) * frame_len
            
            # Distance from frame center to onset time
            d_i = abs(frame_center_time - onset_time)
            
            # Regression target: 1 - abs(d_i) / (J * frame_len)
            if d_i <= J * frame_len:
                target_value = 1.0 - d_i / (J * frame_len)
                # Take maximum if multiple onsets affect the same frame
                onset_frames[frame_idx, piano_key_idx] = max(
                    onset_frames[frame_idx, piano_key_idx], 
                    target_value
                )
        
        # Create regression targets for offset
        for frame_idx in range(max(0, offset_frame - J), min(num_frames, offset_frame + J + 1)):
            # Calculate frame center time
            frame_center_time = (frame_idx + 0.5) * frame_len
            
            # Distance from frame center to offset time
            d_i = abs(frame_center_time - offset_time)
            
            # Regression target: 1 - abs(d_i) / (J * frame_len)
            if d_i <= J * frame_len:
                target_value = 1.0 - d_i / (J * frame_len)
                # Take maximum if multiple offsets affect the same frame
                offset_frames[frame_idx, piano_key_idx] = max(
                    offset_frames[frame_idx, piano_key_idx], 
                    target_value
                )
    
    return {
        'onset': onset_frames,
        'offset': offset_frames,
        'frames': note_frames
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, required=True)
    parser.add_argument('--frames_per_second', type=int, default=100)
    parser.add_argument('--J', type=int, default=5)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()
    
    data_list = pd.read_csv(args.data_list)
    
    for _, row in data_list.iterrows():
        idx = row['example_id']
        midi_path = row['midi_path']
        output_file = os.path.join(args.output_folder, idx + ".frame_labels.npy")
        try:
            labels = extract_frame_labels(midi_path, J=args.J)
            np.save(output_file, labels)
        except Exception as e:
            print(f"Error processing MIDI file: {e}")
            continue
