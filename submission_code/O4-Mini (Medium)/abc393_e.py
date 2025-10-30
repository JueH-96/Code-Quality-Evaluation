import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = list(map(int, it))
    
    # If K == 1, the answer for each i is A[i] itself.
    if K == 1:
        out = sys.stdout.write
        for v in A:
            out(str(v) + "
")
        return

    # Build count of each value
    M = max(A)
    cnt = [0] * (M + 1)
    for v in A:
        cnt[v] += 1

    # freq[d] = number of A_j that are multiples of d
    freq = [0] * (M + 1)
    for d in range(1, M + 1):
        c = 0
        # sum up cnt[m] for m = d, 2d, 3d, ...
        for m in range(d, M + 1, d):
            c += cnt[m]
        freq[d] = c

    # ans[x] = max d dividing x such that freq[d] >= K
    ans = [0] * (M + 1)
    f = freq  # local ref
    a = ans
    # iterate d from 1..M; whenever freq[d] >= K,
    # we know d can be the gcd for any multiple of d
    for d in range(1, M + 1):
        if f[d] >= K:
            # update all multiples
            for m in range(d, M + 1, d):
                a[m] = d

    # output answers for each A_i
    out = sys.stdout.write
    for v in A:
        out(str(a[v]) + "
")

if __name__ == "__main__":
    main()