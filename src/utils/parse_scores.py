from music21 import converter
import music21
import pandas as pd
import os
import numpy as np
import json
import argparse
import pickle

def parse_score(path, file, participant, song):
    all_notes = []
    score = converter.parse(os.path.join(path, file))
    for note in score.recurse().notesAndRests:
        if note.isNote:
            measure = note.getContextByClass('Measure')
            all_notes.append({
                    'participant': participant,
                    'song': song,
                    'pitch': note.pitch,
                    'duration': note.duration.quarterLength,
                    'type': note.duration.type,
                    'position': note.offset,
                    'measure': note.measureNumber,
                    'grace': isinstance(note.duration, music21.duration.GraceDuration),
                    'note': note,
                    'note_type': note.duration.type,
                    'note_dots': note.duration.dots,
                    'components': note.duration.components,
                    'full_name': note.duration.fullName,
                    'tuplets': note.duration.tuplets,
                    'measure_offset': measure.offset,
                })
    return all_notes

def parse_omnibook_scores(path = '../omnibook_stages/raw/CharlieParkerAlignedOmnibookPublication/musicxml'):
    all_notes = []
    files = sorted(os.listdir(path))
    for file in files:
        if not file.endswith('.xml'):
            continue
        all_notes.extend(parse_score(path, file, 'bird', file.split('.xml')[0].split('-')[0]))
    notes_df = pd.DataFrame(all_notes)
    return notes_df

def parse_filosax_scores(path = '../stages/0_copy_scores'):
    all_notes = []
    for participant in range(1, 6):
        for song in range(1, 49):
            filename = f"FS{participant}_{song:02d}.Sax.musicxml"
            all_notes.extend(parse_score(path, filename, participant, song))
    notes_df = pd.DataFrame(all_notes)
    return notes_df

def add_note_onsets_and_durations(notes_df, quarter_len=24):
    notes_df['ticks'] = (notes_df.duration * quarter_len).astype(float).round().astype(int)
    notes_df['onsets'] = (notes_df.position * quarter_len).astype(float).round().astype(int)
    return notes_df

def encode_measures(notes_df, quarter_len=24):
    measures = []
    for participant in notes_df.participant.unique():
        subset = notes_df[notes_df['participant'] == participant]
        for song in subset.song.unique():
            current_song = subset[subset['song'] == song]
            for unique_offset in current_song['measure_offset'].unique():
                content = current_song[current_song['measure_offset'] == unique_offset]
                encoding = np.ones(quarter_len * 4) * 130
                for i, note in content.iterrows():
                    if note['grace'] != True: 
                        encoding[note.onsets : note.onsets + note.ticks] = 129
                        if note.note.tie != None:
                            assert note.note.tie.type in ['start', 'stop', 'continue'] 
                            if note.note.tie.type == 'start':
                                encoding[note.onsets] = int(note.pitch.midi)
                        else:
                            encoding[note.onsets] = int(note.pitch.midi)
                measures.append({
                    'participant': participant,
                    'song': song,
                    'beatwise_score': [
                        encoding[:quarter_len], 
                        encoding[quarter_len:quarter_len*2], 
                        encoding[quarter_len*2:quarter_len*3], 
                        encoding[quarter_len*3:quarter_len*4]
                    ],
                    'measure_offset': unique_offset,
                })
    measures = pd.DataFrame(measures)
    measures['bar_num'] = (measures.measure_offset / 4).astype(int)
    return measures

def parse_syncpoint_file(path, file, participant, song):
    syncpoints = []
    with open(os.path.join(path, file), 'r') as f:
        syncpoints_raw = json.load(f)
    for line in syncpoints_raw:
        if len(line) == 2:
            syncpoints.append({
                'participant': participant,
                'song': song,
                'bar_num': line[0],
                'syncpoint': line[1],
                'flag1': np.nan,
                'flag2': np.nan,
            })
        elif len(line) == 4:
            syncpoints.append({
                'participant': participant,
                'song': song,
                'bar_num': line[0],
                'syncpoint': line[1],
                'flag1': line[2],
                'flag2': line[3],
            })
    return syncpoints

def parse_omnibook_syncpoints(path = '../omnibook_stages/raw/CharlieParkerAlignedOmnibookPublication/syncpoints'):
    syncpoints_files = sorted(os.listdir(path))
    syncpoints = []
    for file in syncpoints_files:
        if not file.endswith('.json'):
            continue
        song_name = file.split('-')[0]
        syncpoints.extend(parse_syncpoint_file(path, file, 'bird', song_name))
    syncpoints = pd.DataFrame(syncpoints)
    return syncpoints

def parse_filosax_syncpoints(path = '../stages/0_copy_scores'):
    syncpoints = []
    for participant in range(1, 6):
        for song in range(1, 49):
            song_name = f'FS{participant}_{song:02d}.Sax-predicted-syncpoints.json'
            syncpoints.extend(parse_syncpoint_file(path, song_name, participant, song))
    syncpoints = pd.DataFrame(syncpoints)
    return syncpoints

def compile_annotations():
    omnibook_scores = parse_omnibook_scores()
    omnibook_scores = add_note_onsets_and_durations(omnibook_scores)
    omnibook_scores = encode_measures(omnibook_scores)
    omnibook_syncpoints = parse_omnibook_syncpoints()
    omnibook_scores = pd.merge(omnibook_scores, omnibook_syncpoints, on=['participant', 'song', 'bar_num'], how='outer')
    
    filosax_scores = parse_filosax_scores()
    filosax_scores = add_note_onsets_and_durations(filosax_scores)
    filosax_scores = encode_measures(filosax_scores)
    filosax_syncpoints = parse_filosax_syncpoints()
    filosax_scores = pd.merge(filosax_scores, filosax_syncpoints, on=['participant', 'song', 'bar_num'], how='outer')
    
    combined_scores = pd.concat([omnibook_scores, filosax_scores])
    return combined_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse music scores from Omnibook and Filosax datasets')
    parser.add_argument('--output', type=str, default='stages/1_processed_scores/combined_scores.pkl',
                       help='Output file path for the processed scores (default: combined_scores.pkl)')
    parser.add_argument('--omnibook-path', type=str, 
                       default='stages/0_raw/CharlieParkerAlignedOmnibookPublication/musicxml',
                       help='Path to Omnibook musicxml files')
    parser.add_argument('--omnibook-syncpoints-path', type=str,
                       default='stages/0_raw/CharlieParkerAlignedOmnibookPublication/syncpoints',
                       help='Path to Omnibook syncpoints files')
    parser.add_argument('--filosax-path', type=str, default='stages/0_raw/filosax_scores',
                       help='Path to Filosax scores')
    parser.add_argument('--filosax-syncpoints-path', type=str, default='stages/0_raw/filosax_scores',
                       help='Path to Filosax syncpoints files')
    
    args = parser.parse_args()
    

    print("Processing Omnibook scores...")
    omnibook_scores = parse_omnibook_scores(args.omnibook_path)
    omnibook_scores = add_note_onsets_and_durations(omnibook_scores)
    omnibook_scores = encode_measures(omnibook_scores)
    omnibook_syncpoints = parse_omnibook_syncpoints(args.omnibook_syncpoints_path)
    omnibook_scores = pd.merge(omnibook_scores, omnibook_syncpoints, on=['participant', 'song', 'bar_num'], how='outer')
    

    print("Processing Filosax scores...")
    filosax_scores = parse_filosax_scores(args.filosax_path)
    filosax_scores = add_note_onsets_and_durations(filosax_scores)
    filosax_scores = encode_measures(filosax_scores)
    filosax_syncpoints = parse_filosax_syncpoints(args.filosax_path)
    filosax_scores = pd.merge(filosax_scores, filosax_syncpoints, on=['participant', 'song', 'bar_num'], how='outer')
    
    final_scores = pd.concat([omnibook_scores, filosax_scores])
    
    print(f"Saving {len(final_scores)} records to {args.output}")
    #final_scores.to_csv(args.output, index=False)
    pickle.dump(final_scores, open(args.output, 'wb'))
    print("Processing complete!")
    