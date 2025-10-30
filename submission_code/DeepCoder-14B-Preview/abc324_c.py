def is_insertion(s, t):
    i = j = 0
    len_s = len(s)
    len_t = len(t)
    if len_s + 1 != len_t:
        return False
    allowed_skips = 1
    while i < len_s and j < len_t:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if allowed_skips:
                j += 1
                allowed_skips -= 1
            else:
                return False
    if i < len_s:
        return False
    if j < len_t:
        if allowed_skips >= (len_t - j):
            return True
        else:
            return False
    else:
        return True

def is_deletion(s, t):
    i = j = 0
    len_s = len(s)
    len_t = len(t)
    if len_s - 1 != len_t:
        return False
    allowed_skips = 1
    while i < len_s and j < len_t:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if allowed_skips:
                i += 1
                allowed_skips -= 1
            else:
                return False
    if j < len_t:
        return False
    if i < len_s:
        if allowed_skips >= (len_s - i):
            return True
        else:
            return False
    else:
        return True

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T_prime = input[ptr]
    ptr += 1
    S_list = input[ptr:ptr + N]
    ptr += N

    res = []
    for idx in range(1, N + 1):
        S = S_list[idx - 1]
        len_S = len(S)
        len_T = len(T_prime)
        if len_S == len_T:
            sum_diff = sum(a != b for a, b in zip(S, T_prime))
            if sum_diff <= 1:
                res.append(idx)
        elif len_S == len_T - 1:
            if is_insertion(S, T_prime):
                res.append(idx)
        elif len_S == len_T + 1:
            if is_deletion(S, T_prime):
                res.append(idx)
    
    print(len(res))
    if res:
        print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()