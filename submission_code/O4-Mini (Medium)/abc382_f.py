import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    H,W,N = map(int, input().split())
    bars = []
    for i in range(N):
        r,c,l = map(int, input().split())
        # store (initial_row, left, right_excl, original_index)
        bars.append((r, c-1, c-1 + l, i))

    # Sort in descending order of initial row
    bars.sort(key=lambda x: -x[0])

    INF = 10**18
    # We'll build a segment tree of size W, 0-based, storing:
    #   max1: maximum in the interval
    #   smax: second maximum
    #   cnt: how many times max1 occurs
    #   min1: minimum in the interval
    size = 1
    while size < W: size <<= 1

    max1 = [0]*(2*size)
    smax = [ -INF ]*(2*size)
    cnt  = [1]*(2*size)
    min1 = [0]*(2*size)

    # Initialize all leaves to H+1
    for i in range(W):
        idx = size + i
        max1[idx] = H+1
        min1[idx] = H+1
        smax[idx] = -INF
        cnt[idx]  = 1
    for i in range(W, size):
        idx = size + i
        max1[idx] = -INF
        min1[idx] = INF
        smax[idx] = -INF
        cnt[idx]  = 0

    # Build internal nodes
    for p in range(size-1, 0, -1):
        left, right = 2*p, 2*p+1
        # Merge min
        min1[p] = min(min1[left], min1[right])
        # Merge max1, smax, cnt
        if max1[left] > max1[right]:
            max1[p] = max1[left]
            cnt[p]  = cnt[left]
            smax[p] = max(smax[left], max1[right])
        elif max1[left] < max1[right]:
            max1[p] = max1[right]
            cnt[p]  = cnt[right]
            smax[p] = max(smax[right], max1[left])
        else:
            max1[p] = max1[left]
            cnt[p]  = cnt[left] + cnt[right]
            smax[p] = max(smax[left], smax[right])

    # Helper to push a cap x = "chmin" onto node p
    def apply_chmin(p, x):
        if max1[p] <= x:
            return
        max1[p] = x
        # min1 is unaffected, since x >= old_second_max >= old_min
        # smax remains the same or below x, no change needed

    # Push a node's caps down to its children
    def push(p):
        for c in (2*p, 2*p+1):
            if max1[c] > max1[p]:
                apply_chmin(c, max1[p])

    # Recompute node p from its children
    def pull(p):
        left, right = 2*p, 2*p+1
        min1[p] = min(min1[left], min1[right])
        if max1[left] > max1[right]:
            max1[p] = max1[left]
            cnt[p]  = cnt[left]
            smax[p] = max(smax[left], max1[right])
        elif max1[left] < max1[right]:
            max1[p] = max1[right]
            cnt[p]  = cnt[right]
            smax[p] = max(smax[right], max1[left])
        else:
            max1[p] = max1[left]
            cnt[p]  = cnt[left] + cnt[right]
            smax[p] = max(smax[left], smax[right])

    # Range chmin in [L,R)
    def range_chmin(L, R, x, p=1, l=0, r=size):
        if r <= L or R <= l or max1[p] <= x:
            return
        if L <= l and r <= R and smax[p] < x < max1[p]:
            # we can cap the entire node
            apply_chmin(p, x)
            return
        # otherwise push down and recurse
        push(p)
        m = (l+r)//2
        range_chmin(L, R, x, 2*p,   l,   m)
        range_chmin(L, R, x, 2*p+1, m,   r)
        pull(p)

    # Range min query in [L,R)
    def range_min(L, R, p=1, l=0, r=size):
        if r <= L or R <= l:
            return INF
        if L <= l and r <= R:
            return min1[p]
        push(p)
        m = (l+r)//2
        return min(range_min(L,R,2*p,  l,   m),
                   range_min(L,R,2*p+1,m,   r))

    ans = [0]*N

    # Process bars from lowest (largest R) to highest
    for R, L, RR, idx in bars:
        # Query how far we can drop: look at min support in [L..RR)
        s = range_min(L,RR)
        d = s - R - 1
        Rf = R + d
        ans[idx] = Rf
        # Now those columns are blocked at row Rf
        range_chmin(L, RR, Rf)

    # Output in original order
    out = sys.stdout
    for v in ans:
        out.write(str(v) + "
")

if __name__ == "__main__":
    main()