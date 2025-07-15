import argparse
import pretty_midi
import torch
import numpy as np
import soundfile as sf
import librosa

def prepare_beats(beats):
    syncpoints = beats[:,0]
    beat_nums = beats[:,1]
    bars = []
    content = []
    for i in range(len(syncpoints)):
        if beat_nums[i] == 1:
            if len(content) == 4:
                content.append(syncpoints[i]) # that's right, should be 1, 2, 3, 4, 1
                bars.append(content)
            content = []
        content.append(syncpoints[i])
    return bars


def tokens_to_score_midi(tokens, 
                  tempo: float = 120.0,
                  add_clicks: bool = True,
                  click_type: str = 'woodblock',
                  compensate = 0) -> pretty_midi.PrettyMIDI:
    """
    Convert token sequence to MIDI file.
    Assumes batched input of consecutive 8-bar 4/4 fragments with 48 time steps per bar.
    
    Args:
        tokens: Token sequence of shape [N, 8, 48] or list of tokens, where N is number of consecutive slices
        tempo: Tempo in BPM (default: 120)
        ticks_per_beat: MIDI ticks per beat (default: 480)
        add_clicks: Whether to add click tracks (default: True)
        click_type: Type of click sound ('woodblock' or 'claves', default: 'woodblock')
        
    Returns:
        pretty_midi.PrettyMIDI: MIDI file object
    """
    # Convert input to numpy array if it's a tensor
    tokens = tokens.reshape(-1, 48)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    elif isinstance(tokens, list):
        tokens = np.array(tokens)
    
    
    # Create MIDI file
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano_program = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano
    
    # Create click track if requested
    if add_clicks:
        if click_type == 'woodblock':
            click_program = pretty_midi.Instrument(program=115)  # 115 = Woodblock
            downbeat_pitch = 84
            beat_pitch = 76
        else:  # claves
            click_program = pretty_midi.Instrument(program=75)  # 75 = Claves
            downbeat_pitch = 84
            beat_pitch = 76
    
    # Convert tempo to seconds per beat
    seconds_per_beat = 60.0 / tempo
    
    # Process each bar in the slice
    current_pitch = None
    current_start = None
    current_duration = 0
    for bar_idx in range(len(tokens)):
        global_bar_idx = bar_idx
        
        # Add clicks if requested
        if add_clicks:
            # Add downbeat click (higher)
            downbeat_time = global_bar_idx * 4 * seconds_per_beat
            click = pretty_midi.Note(
                velocity=100,
                pitch=downbeat_pitch,
                start=downbeat_time,
                end=downbeat_time + 0.05
            )
            click_program.notes.append(click)
            
            # Add other beat clicks (lower)
            for beat in range(1, 4):
                beat_time = (global_bar_idx * 4 + beat) * seconds_per_beat
                click = pretty_midi.Note(
                    velocity=80,
                    pitch=beat_pitch,
                    start=beat_time,
                    end=beat_time + 0.05
                )
                click_program.notes.append(click)
            
            # Process each time step in the bar
            for t in range(48):
                token = tokens[bar_idx, t]
                
                if token == 129:  # Rest token
                    if current_pitch is not None:  # End current note
                        start_time = (current_start) * seconds_per_beat / 12
                        end_time = (current_start + current_duration) * seconds_per_beat / 12
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=current_pitch - compensate,
                            start=start_time,
                            end=end_time
                        )
                        piano_program.notes.append(note)
                        current_pitch = None
                        current_start = None
                        current_duration = 0
                elif token == 128:  # Tie token
                    if current_pitch is not None:  # Extend current note
                        current_duration += 1
                else:  # Note token (1-127)
                    if current_pitch is None:  # New note
                        current_pitch = token
                        current_start = global_bar_idx * 48 + t
                        current_duration = 1
                    elif token == current_pitch:  # Same note continues
                        current_duration += 1
                    else:  # Different note
                        # Add previous note to MIDI
                        if current_pitch is not None:
                            start_time = (current_start) * seconds_per_beat / 12
                            end_time = (current_start + current_duration) * seconds_per_beat / 12
                            note = pretty_midi.Note(
                                velocity=100,
                                pitch=current_pitch - compensate,
                                start=start_time,
                                end=end_time
                            )
                            piano_program.notes.append(note)
                        
                        # Start new note
                        current_pitch = token
                        current_start = global_bar_idx * 48 + t
                        current_duration = 1
            
    # Handle last note
    if current_pitch is not None:
        start_time = (current_start) * seconds_per_beat / 12
        end_time = (current_start + current_duration) * seconds_per_beat / 12
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_pitch - compensate,
            start=start_time,
            end=end_time
        )
        piano_program.notes.append(note)
    
    # Add instruments to MIDI file
    midi.instruments.append(piano_program)
    if add_clicks:
        midi.instruments.append(click_program)
    
    return midi



def tokens_to_performance_midi(tokens, 
                  tempo: float = 120.0,
                  ticks_per_beat: int = 480,
                  add_clicks: bool = True,
                  click_type: str = 'woodblock',
                  beats = None, 
                  compensate = 0) -> pretty_midi.PrettyMIDI:
    """
    Convert token sequence to MIDI file.
    Assumes batched input of consecutive 8-bar 4/4 fragments with 48 time steps per bar.
    
    Args:
        tokens: Token sequence of shape [N, 8, 48] or list of tokens, where N is number of consecutive slices
        tempo: Tempo in BPM (default: 120)
        ticks_per_beat: MIDI ticks per beat (default: 480)
        add_clicks: Whether to add click tracks (default: True)
        click_type: Type of click sound ('woodblock' or 'claves', default: 'woodblock')
        
    Returns:
        pretty_midi.PrettyMIDI: MIDI file object
    """
    # Convert input to numpy array if it's a tensor
    tokens = tokens.reshape(-1, 48)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    elif isinstance(tokens, list):
        tokens = np.array(tokens)

    
    
    # Create MIDI file
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano
    
    # Create click track if requested
    if add_clicks:
        if click_type == 'woodblock':
            click_program = pretty_midi.Instrument(program=115)  # 115 = Woodblock
            downbeat_pitch = 84
            beat_pitch = 76
        else:  # claves
            click_program = pretty_midi.Instrument(program=75)  # 75 = Claves
            downbeat_pitch = 84
            beat_pitch = 76
    
    # Convert tempo to seconds per beat
    
        
    # Process each bar in the slice
    current_pitch = None
    current_start = None
    current_duration = 0    
    for bar_idx in range(len(tokens)):
        
        
        # Add clicks if requested
        if add_clicks:
            # Add downbeat click (higher)
            downbeat_time = beats[bar_idx][0]
            click = pretty_midi.Note(
                velocity=100,
                pitch=downbeat_pitch,
                start=downbeat_time,
                end=downbeat_time + 0.05
            )
            click_program.notes.append(click)
            
            # Add other beat clicks (lower)
            for beat in range(1, 4):
                beat_time = beats[bar_idx][beat]
                click = pretty_midi.Note(
                    velocity=80,
                    pitch=beat_pitch,
                    start=beat_time,
                    end=beat_time + 0.05
                )
                click_program.notes.append(click)
        
        # Process each time step in the bar
        for t in range(48):
            if t in range(12):
                seconds_per_beat = (beats[bar_idx][1] - beats[bar_idx][0])
            elif t in range(12,24):
                seconds_per_beat = (beats[bar_idx][2] - beats[bar_idx][1])
            elif t in range(24,36):
                seconds_per_beat = (beats[bar_idx][3] - beats[bar_idx][2])
            elif t in range(46,48):
                seconds_per_beat = (beats[bar_idx][2] - beats[bar_idx][1])
            token = tokens[bar_idx, t]
            
            if token == 129:  # Rest token
                if current_pitch is not None:  # End current note
                    start_time = current_start 
                    end_time = current_start + current_duration
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=current_pitch - compensate,
                        start=start_time,
                        end=end_time
                    )
                    piano_program.notes.append(note)
                    current_pitch = None
                    current_start = None
                    current_duration = 0
            elif token == 128:  # Tie token
                if current_pitch is not None:  # Extend current note
                    current_duration += seconds_per_beat / 12
            else:  # Note token (1-127)
                if current_pitch is None:  # New note
                    current_pitch = token
                    current_start = beats[bar_idx][0] + t * seconds_per_beat /12
                    current_duration = seconds_per_beat / 12
                elif token == current_pitch:  # Same note continues
                    current_duration += seconds_per_beat / 12
                else:  # Different note
                    # Add previous note to MIDI
                    if current_pitch is not None:
                        start_time = current_start 
                        end_time = current_start + current_duration
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=current_pitch - compensate,
                            start=start_time,
                            end=end_time
                        )
                        piano_program.notes.append(note)
                    
                    # Start new note
                    current_pitch = token
                    current_start = beats[bar_idx][0] +t * seconds_per_beat / 12
                    current_duration = seconds_per_beat / 12
        
    # Handle last note 
    if current_pitch is not None:
        start_time = current_start
        end_time = current_start + current_duration
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_pitch - compensate,
            start=start_time,
            end=end_time
        )
        piano_program.notes.append(note)
    
    # Add instruments to MIDI file
    midi.instruments.append(piano_program)
    if add_clicks:
        midi.instruments.append(click_program)
    
    return midi

def midi_to_audio_stereo(midi_path, audio_path, output_path, sr=22050):
    # Load MIDI and synthesize using default (sinusoidal) synth
    midi = pretty_midi.PrettyMIDI(midi_path)
    midi_audio = midi.synthesize(fs=sr)  # Pure-Python fallback

    # Load real audio
    audio, _ = librosa.load(audio_path, sr=sr)

    # Pad both to the same length
    max_len = max(len(midi_audio), len(audio))
    midi_audio = np.pad(midi_audio, (0, max_len - len(midi_audio)))
    audio = np.pad(audio, (0, max_len - len(audio)))

    # Combine: MIDI to left, original audio to right
    stereo = np.stack([midi_audio, audio], axis=0)

    # Save as stereo .wav
    sf.write(output_path, stereo.T, sr)
    print(f"Stereo file saved to: {output_path}")

def main(beats_file, tokens, midi_performance_output_file, midi_score_output_file, audio_path, output_path):
    beats = np.loadtxt(beats_file)
    bars = prepare_beats(beats)
    midi = tokens_to_performance_midi(tokens, beats = bars, add_clicks = False)
    midi.write(midi_performance_output_file)
    midi_to_audio_stereo(midi_performance_output_file, audio_path, output_path)
    midi = tokens_to_score_midi(tokens, add_clicks = True)
    midi.write(midi_score_output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beats_file', type=str, required=True)
    parser.add_argument('--tokens_file', type=str, required=True)
    parser.add_argument('--midi_performance_output_file', type=str, required=True)
    parser.add_argument('--midi_score_output_file', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)


    args = parser.parse_args()
    tokens = torch.load(args.tokens_file)
    main(args.beats_file, tokens, args.midi_performance_output_file, args.midi_score_output_file, args.audio_path, args.output_path)
