import sys
data = sys.stdin.read().split()
S = data[0]
Q = int(data[1])
K_list = [int(x) for x in data[2:2+Q]]
L = len(S)

def get_char_at_position(m_start, pos, len_s, s_str):
    swap_parity = 0
    current_pos = pos
    m = m_start
    while m > 0:
        len_first_half = len_s * (1 << (m - 1))
        if current_pos > len_first_half:
            swap_parity = 1 - swap_parity
            current_pos -= len_first_half
        m -= 1
    char = s_str[current_pos - 1]
    if swap_parity == 1:
        char = char.swapcase()
    return char

m_start = 60
result = []
for K in K_list:
    char = get_char_at_position(m_start, K, L, S)
    result.append(char)
print(' '.join(result))