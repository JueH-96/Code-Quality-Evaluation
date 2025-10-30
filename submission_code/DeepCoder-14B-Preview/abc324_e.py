import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = input[ptr]
    ptr += 1
    S = input[ptr:ptr + N]
    ptr += N

    len_t = len(T)
    pre = []
    for s in S:
        t_ptr = 0
        for c in s:
            if t_ptr < len_t and c == T[t_ptr]:
                t_ptr += 1
        pre.append(t_ptr)

    T_pos = {}
    for i, c in enumerate(T):
        if c not in T_pos:
            T_pos[c] = []
        T_pos[c].append(i)

    cnt = [0] * (len_t + 1)

    for j in range(N):
        s = S[j]
        suf = [0] * len_t
        for k in range(len_t):
            t_ptr = k
            count = 0
            for c in s:
                if t_ptr >= len_t:
                    break
                if c == T[t_ptr]:
                    count += 1
                    t_ptr += 1
            suf[k] = count

        any_ok = [False] * (len_t + 1)
        current_ok = False
        for k in range(len_t + 1):
            if k <= len_t - 1:
                if suf[k] >= (len_t - k):
                    current_ok = True
            else:
                if current_ok:
                    pass  # since any_ok[len_t] would already be True if any previous was True
            if k == 0:
                any_ok[k] = (suf[0] >= (len_t - 0)) if len_t > 0 else False
            else:
                if k < len_t:
                    any_ok[k] = any_ok[k-1] or (suf[k] >= (len_t - k))
                else:
                    any_ok[k] = any_ok[k-1]
        for k in range(len_t + 1):
            if any_ok[k]:
                if k <= len_t:
                    cnt[k] += 1

    total = 0
    for p in pre:
        if p <= len_t:
            total += cnt[p]
    print(total)

if __name__ == '__main__':
    main()