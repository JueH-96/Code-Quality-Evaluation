import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute prefix distinct counts
    pre = [0] * (N + 1)
    seen = [False] * (N + 1)
    cnt = 0
    for i in range(1, N + 1):
        x = A[i-1]
        if not seen[x]:
            seen[x] = True
            cnt += 1
        pre[i] = cnt

    # Compute suffix distinct counts
    suf = [0] * (N + 2)
    seen = [False] * (N + 1)
    cnt = 0
    for i in range(N, 0, -1):
        x = A[i-1]
        if not seen[x]:
            seen[x] = True
            cnt += 1
        suf[i] = cnt

    # Segment Tree for range add and range max
    size = 1
    while size < N + 2:
        size <<= 1
    # We'll use a 1-based segment tree over [1..N], size is next power of two
    segsize = 2 * size
    tree = [0] * segsize
    lazy = [0] * segsize

    # Build initial tree leaves at [size .. size+N]
    # We put pre[i] at position size + i
    for i in range(1, N + 1):
        tree[size + i] = pre[i]
    # Build internal nodes
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[2*i], tree[2*i+1])

    # Push-down utility
    def apply(node, val):
        tree[node] += val
        lazy[node] += val

    def push(node):
        v = lazy[node]
        if v:
            apply(node*2, v)
            apply(node*2+1, v)
            lazy[node] = 0

    # Range update [l..r] add val
    def update(node, nl, nr, l, r, val):
        if l <= nl and nr <= r:
            tree[node] += val
            lazy[node] += val
            return
        if nr < l or r < nl:
            return
        push(node)
        mid = (nl + nr) // 2
        if l <= mid:
            update(node*2, nl, mid, l, r, val)
        if r > mid:
            update(node*2+1, mid+1, nr, l, r, val)
        tree[node] = max(tree[node*2], tree[node*2+1])

    # Range query max on [l..r]
    def query(node, nl, nr, l, r):
        if l <= nl and nr <= r:
            return tree[node]
        if nr < l or r < nl:
            return -10**18
        push(node)
        mid = (nl + nr) // 2
        res = -10**18
        if l <= mid:
            res = query(node*2, nl, mid, l, r)
        if r > mid:
            v = query(node*2+1, mid+1, nr, l, r)
            if v > res:
                res = v
        return res

    lastpos = [0] * (N + 1)
    ans = 0

    # Iterate j from 2 to N-1
    for j in range(2, N):
        x = A[j-1]
        p = lastpos[x]
        # valid i are in [1..j-1], and delta +1 for i >= p
        l = p if p > 0 else 1
        if l <= j-1:
            update(1, 1, size, l, j-1, 1)
        # f_j = max over i in [1..j-1]
        best = query(1, 1, size, 1, j-1)
        # add suffix starting j+1
        total = best + suf[j+1]
        if total > ans:
            ans = total
        lastpos[x] = j

    print(ans)

if __name__ == "__main__":
    main()