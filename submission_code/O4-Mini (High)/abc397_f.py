import sys
def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # prev[j] = previous index of A[j], or -1 if none
    last = {}
    prev = [-1] * N
    for i, v in enumerate(A):
        p = last.get(v, -1)
        prev[i] = p
        last[v] = i

    # L[i] = distinct count in A[0..i]
    seen = [False] * (N + 1)
    L = [0] * N
    cnt = 0
    for i, v in enumerate(A):
        if not seen[v]:
            seen[v] = True
            cnt += 1
        L[i] = cnt

    # R[i] = distinct count in A[i..N-1]
    seen2 = [False] * (N + 1)
    R = [0] * N
    cnt2 = 0
    for i in range(N-1, -1, -1):
        v = A[i]
        if not seen2[v]:
            seen2[v] = True
            cnt2 += 1
        R[i] = cnt2

    # Build segment tree for range-add and range-max
    size = 1
    while size < N:
        size <<= 1
    n = size
    H = n.bit_length() - 1
    NEG_INF = -10**18

    # t: segment-tree nodes, size 2*n
    # lazy: pending adds for internal nodes, size n
    t = [NEG_INF] * (2 * n)
    lazy = [0] * n

    # initialize leaves with L[i]
    for i in range(N):
        t[n + i] = L[i]
    # rest of leaves remain NEG_INF (unused)
    # build upward
    for i in range(n-1, 0, -1):
        # max of children
        a = t[i<<1]; b = t[(i<<1)|1]
        t[i] = a if a > b else b

    # apply add 'val' to node p
    def apply(p, val, t=t, lazy=lazy, n=n):
        t[p] += val
        if p < n:
            lazy[p] += val

    # push down all pending adds on path to p
    def push(p, t=t, lazy=lazy, n=n, H=H):
        for h in range(H, 0, -1):
            i = p >> h
            v = lazy[i]
            if v:
                # children
                lch = i << 1
                rch = lch | 1
                t[lch] += v
                if lch < n:
                    lazy[lch] += v
                t[rch] += v
                if rch < n:
                    lazy[rch] += v
                lazy[i] = 0

    # rebuild nodes on path from p up to root
    def rebuild(p, t=t, lazy=lazy, n=n):
        while p > 1:
            p //= 2
            lch = p << 1
            rch = lch | 1
            a = t[lch]; b = t[rch]
            # take max child + pending
            t[p] = (a if a > b else b) + lazy[p]

    # range add +val on [l..r] (0-based, inclusive)
    def range_add(l, r, val,
                  apply=apply, push=push, rebuild=rebuild,
                  n=n):
        if l > r:
            return
        l += n; r += n
        l0, r0 = l, r
        push(l0); push(r0)
        while l <= r:
            if (l & 1) == 1:
                apply(l, val)
                l += 1
            if (r & 1) == 0:
                apply(r, val)
                r -= 1
            l //= 2; r //= 2
        rebuild(l0); rebuild(r0)

    # range max on [l..r]
    def range_max(l, r,
                  push=push, t=t, n=n, NEG_INF=NEG_INF):
        if l > r:
            return NEG_INF
        l += n; r += n
        push(l); push(r)
        res = NEG_INF
        while l <= r:
            if (l & 1) == 1:
                v = t[l]
                if v > res: res = v
                l += 1
            if (r & 1) == 0:
                v = t[r]
                if v > res: res = v
                r -= 1
            l //= 2; r //= 2
        return res

    # Main loop: try split points j = end of middle segment
    ans = 0
    # for j from 1 to N-2 inclusive (0-based)
    for j in range(1, N-1):
        p = prev[j]
        l = p if p >= 0 else 0
        r = j - 1
        # adding A[j] extends middle segment: for i>=p, D(i+1..j) gains +1
        range_add(l, r, 1)
        # now best[i] = L[i] + D(i+1..j); pick max for i in [0..j-1]
        gm = range_max(0, j-1)
        # add suffix distinct count from j+1..N-1
        cand = gm + R[j+1]
        if cand > ans:
            ans = cand

    print(ans)

if __name__ == "__main__":
    main()