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