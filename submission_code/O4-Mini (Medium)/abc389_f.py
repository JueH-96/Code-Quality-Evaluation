import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline
    N = int(input())
    LR = [tuple(map(int, input().split())) for _ in range(N)]
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]
    # Maximum coordinate
    M = 500000
    # Next power of two ≥ M
    P = 1
    while P < M:
        P <<= 1
    size = 2 * P
    # Segment tree storing max and lazy
    maxv = [0] * size
    lazy = [0] * size
    # Initialize leaves
    for i in range(1, M+1):
        maxv[P + i - 1] = i
    for i in range(M+1, P+1):
        maxv[P + i - 1] = 10**18  # large, so won't trigger
    # Build
    for k in range(P-1, 0, -1):
        maxv[k] = max(maxv[2*k], maxv[2*k+1])
    # Helpers
    def apply(k, v):
        maxv[k] += v
        lazy[k] += v
    def push(k):
        v = lazy[k]
        if v:
            apply(k*2, v)
            apply(k*2+1, v)
            lazy[k] = 0
    # Range add [a,b]
    def _add(a, b, v, k, l, r):
        if b < l or r < a:
            return
        if a <= l and r <= b:
            apply(k, v)
            return
        push(k)
        m = (l + r) >> 1
        _add(a, b, v, 2*k, l, m)
        _add(a, b, v, 2*k+1, m+1, r)
        maxv[k] = max(maxv[2*k], maxv[2*k+1])
    # Find first x in [1,M] with A[x] >= L
    def find_first_ge(L):
        if maxv[1] < L:
            return M+1
        k = 1; l = 1; r = P
        while k < P:
            push(k)
            m = (l + r) >> 1
            if maxv[2*k] >= L:
                k = 2*k
                r = m
            else:
                k = 2*k+1
                l = m+1
        return l if l <= M else M+1
    # Find first x in [1,M] with A[x] > R
    def find_first_gt(R):
        if maxv[1] <= R:
            return M+1
        k = 1; l = 1; r = P
        while k < P:
            push(k)
            m = (l + r) >> 1
            if maxv[2*k] > R:
                k = 2*k
                r = m
            else:
                k = 2*k+1
                l = m+1
        return l if l <= M else M+1
    # Apply all intervals
    for (L, R) in LR:
        lo = find_first_ge(L)
        if lo > M:
            continue
        hi_pos = find_first_gt(R) - 1
        if hi_pos < lo:
            continue
        _add(lo, hi_pos, 1, 1, 1, P)
    # Point query
    def query(pos):
        k = 1; l = 1; r = P
        while k < P:
            push(k)
            m = (l + r) >> 1
            if pos <= m:
                k = 2*k
                r = m
            else:
                k = 2*k+1
                l = m+1
        return maxv[k]
    # Answer queries
    out = []
    for x in queries:
        out.append(str(query(x)))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()