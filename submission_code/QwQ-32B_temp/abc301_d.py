import sys

def compute_max(S, N):
    L = len(S)
    N_bin = bin(N)[2:]
    M = len(N_bin)
    if L != M:
        return -1
    res = 0
    is_tight = True
    for i in range(L):
        s_bit = S[i]
        n_bit = int(N_bin[i])
        possible = []
        if s_bit == '0':
            possible = [0]
        elif s_bit == '1':
            possible = [1]
        else:
            possible = [0, 1]
        allowed = []
        if is_tight:
            allowed = [b for b in possible if b <= n_bit]
            if not allowed:
                return -1
            chosen = max(allowed)
            new_tight = (chosen == n_bit)
        else:
            chosen = 1
            new_tight = False
        res += chosen * (1 << (L - 1 - i))
        is_tight = new_tight
    return res

def main():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    M = len(bin(N)) - 2
    L = len(S)
    if L > M:
        valid = True
        for i in range(L - M):
            c = S[i]
            if c == '1':
                valid = False
                break
        if not valid:
            print(-1)
        else:
            new_S = S[L - M:]
            res = compute_max(new_S, N)
            print(res if res != -1 else -1)
    elif L < M:
        max_val = 0
        for c in S:
            if c == '?':
                bit = 1
            else:
                bit = int(c)
            max_val = max_val * 2 + bit
        print(max_val)
    else:
        res = compute_max(S, N)
        print(res if res != -1 else -1)

if __name__ == "__main__":
    main()