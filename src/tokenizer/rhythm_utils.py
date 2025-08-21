from .rhythm_tokens import RHYTHM_TOKENS

def rhythm_str_to_int(quaternary_str: str) -> int:
    substitution = {'o': '0', 't': '1', 'r': '2', 'x': '3'}
    quaternary_str = quaternary_str.translate(str.maketrans(substitution))
    return int(quaternary_str, base=4)

def int_to_rhythm_str(integer: int) -> str:
    result = 'oooooooooooo'
    if integer != 0:
        substitution = {0: 'o', 1: 't', 2: 'r', 3: 'x'}
        digits = []
        while integer:
            digits.append(int(integer % 4))
            integer //= 4
        str_end = ''.join([substitution[digit] for digit in digits[::-1]])
        result = result[:-len(str_end)] + str_end
    return result

def signature_to_mask(signature: str):
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(signature)):
        if signature[i] == 'o':
            mask[i] = 0
        elif signature[i] == 't':
            mask[i] = 1
        elif signature[i] == 'r':
            mask[i] = 2
        elif signature[i] == 'x':
            mask[i] = 3
    return mask

def generate_inferred_time_feel(signature: str):
    if signature in RHYTHM_TOKENS:
        if RHYTHM_TOKENS[signature]['feel'] == 'swing':
            return 0
        elif RHYTHM_TOKENS[signature]['feel'] == 'double_time':
            return 1
        else:
            assert False
    elif 'x' in signature:
        return 1
    else:
        return 1 # too rare rhythm 

def generate_rhythm_token(signature: str):
    if signature in RHYTHM_TOKENS:
        return RHYTHM_TOKENS[signature]['id']
    elif 'x' in signature:
        return 42 # too fast rhythm
    else:
        return 43 # too rare rhythm
