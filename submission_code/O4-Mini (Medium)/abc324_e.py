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

    # Compute A[i]: how many prefix of T can S[i] match as subsequence
    A = [0] * N
    for idx, s in enumerate(Ss):
        p = 0
        # match T[0], T[1], ...
        for ch in s:
            if p < M and ch == T[p]:
                p += 1
                if p == M:
                    break
        A[idx] = p

    # Compute B[j]: how many suffix of T can S[j] match as subsequence
    # i.e. longest prefix match with reversed T in reversed S
    Tr = T[::-1]
    B = [0] * N
    for idx, s in enumerate(Ss):
        q = 0
        # reversed scan
        for ch in reversed(s):
            if q < M and ch == Tr[q]:
                q += 1
                if q == M:
                    break
        B[idx] = q

    # Sort B for binary search
    B.sort()
    from bisect import bisect_left

    total = 0
    # For each i, count j such that A[i] + B[j] >= M
    # => B[j] >= M - A[i]
    for a in A:
        need = M - a
        # find first j with B[j] >= need
        # if need <= 0, bisect_left returns 0
        idx = bisect_left(B, need)
        total += (N - idx)

    # Output
    sys.stdout.write(str(total))

if __name__ == "__main__":
    main()