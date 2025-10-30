import sys
import threading

def main():
    import sys

    data = sys.stdin
    line = data.readline().split()
    N = int(line[0])
    T = line[1].strip()
    M = len(T)

    Ss = [data.readline().strip() for _ in range(N)]

    # Compute pre_i: max prefix of T matched as a subsequence in S_i
    pre = [0] * N
    for idx, S in enumerate(Ss):
        p = 0
        # match T[0..] in S
        for ch in S:
            if p < M and ch == T[p]:
                p += 1
                if p == M:
                    break
        pre[idx] = p

    # Compute suf_j: max suffix of T matched as subsequence in S_j
    # by reversing strings and T, then same as prefix match
    RT = T[::-1]
    suf = [0] * N
    for idx, S in enumerate(Ss):
        p = 0
        # match RT[0..] in reversed S
        for ch in reversed(S):
            if p < M and ch == RT[p]:
                p += 1
                if p == M:
                    break
        suf[idx] = p

    # Count frequencies of suf values
    # suf values are between 0 and M inclusive
    freqB = [0] * (M + 1)
    for v in suf:
        freqB[v] += 1

    # Build suffix sums: suff_count[t] = sum_{k>=t} freqB[k]
    suff_count = [0] * (M + 2)
    acc = 0
    for k in range(M, -1, -1):
        acc += freqB[k]
        suff_count[k] = acc

    # For each pre_i, we need count of j with suf_j >= M - pre_i
    ans = 0
    for p in pre:
        need = M - p
        if need <= 0:
            # any j works
            ans += N
        elif need <= M:
            ans += suff_count[need]
        # else need > M => no suf_j can satisfy

    # Output
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()