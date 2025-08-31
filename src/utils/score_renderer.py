from music21 import stream, note, meter, tempo


def ints_to_musicxml(tokens, out_fp="out.musicxml", bpm=120):

    BINS_PER_BEAT = 12
    QL_PER_BIN = 1 / BINS_PER_BEAT  # quarterLength per bin
    ONSET_MAX = 127
    TIE = 128
    REST = 129

    """
    tokens: list[int] where 0..127=<ONSET X>, 128=<TIE>, 129=<REST>
    12 bins per beat, always 4/4, monophonic.
    """
    part = stream.Part()
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, tempo.MetronomeMark(number=bpm))

    cur_kind = None   # 'note' | 'rest' | None
    cur_pitch = None
    cur_len_bins = 0

    def flush():
        nonlocal cur_kind, cur_pitch, cur_len_bins
        if cur_len_bins == 0:
            return
        ql = cur_len_bins * QL_PER_BIN
        if cur_kind == 'rest':
            obj = note.Rest(quarterLength=ql)
        else:
            obj = note.Note(midi=cur_pitch, quarterLength=ql)
        part.append(obj)
        cur_kind = None
        cur_pitch = None
        cur_len_bins = 0

    for tok in tokens:
        if 0 <= tok <= ONSET_MAX:            # <ONSET X>
            flush()
            cur_kind = 'note'
            cur_pitch = tok
            cur_len_bins = 1
        elif tok == TIE:                      # extend current event
            if cur_kind is None:
                # treat stray TIE as rest extension (robust fallback)
                cur_kind = 'rest'
            cur_len_bins += 1
        elif tok == REST:
            if cur_kind == 'rest':
                cur_len_bins += 1
            else:
                flush()
                cur_kind = 'rest'
                cur_pitch = None
                cur_len_bins = 1
        else:
            raise ValueError(f"Bad token: {tok}")

    flush()

    scored = part.makeMeasures(inPlace=False)  # split to bars
    scored.makeNotation(inPlace=True)          # add ties over barlines, etc.
    scored.write('musicxml', fp=out_fp)
    return out_fp