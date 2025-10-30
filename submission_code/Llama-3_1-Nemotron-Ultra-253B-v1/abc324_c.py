import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    T_prime = data[ptr]
    ptr += 1
    S_list = data[ptr:ptr+N]
    ptr += N

    next_pos = precompute_next_pos(T_prime)
    t_len = len(T_prime)

    result = []

    for idx in range(N):
        s = S_list[idx]
        s_len = len(s)
        cond = False

        if s == T_prime:
            cond = True
        else:
            if s_len == t_len:
                diff = 0
                for a, b in zip(s, T_prime):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    cond = True
            if not cond and s_len == t_len - 1:
                cond = is_subsequence(s, next_pos, t_len)
            if not cond and s_len == t_len + 1:
                cond = is_subsequence_t_in_s(s, T_prime)
        
        if cond:
            result.append(idx + 1)

    print(len(result))
    if result:
        print(' '.join(map(str, sorted(result))))

def precompute_next_pos(s):
    n = len(s)
    next_pos = [[n] * 26 for _ in range(n + 1)]
    last = [n] * 26
    for i in range(n - 1, -1, -1):
        c = ord(s[i]) - ord('a')
        next_pos[i] = last.copy()
        last[c] = i
    return next_pos

def is_subsequence(s, next_pos, t_len):
    pos = 0
    for c_char in s:
        if pos >= t_len:
            return False
        c = ord(c_char) - ord('a')
        next_p = next_pos[pos][c]
        if next_p >= t_len:
            return False
        pos = next_p + 1
    return True

def is_subsequence_t_in_s(s, t):
    t_len = len(t)
    s_len = len(s)
    if t_len > s_len:
        return False
    i = 0
    j = 0
    while i < t_len and j < s_len:
        if t[i] == s[j]:
            i += 1
        j += 1
    return i == t_len

if __name__ == '__main__':
    main()