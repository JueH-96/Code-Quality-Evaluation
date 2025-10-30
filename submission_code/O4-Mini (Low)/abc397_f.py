import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # Compute L[i]: distinct count in prefix 1..i
    seen = set()
    L = [0]* (N+1)
    cnt = 0
    for i in range(1, N+1):
        if A[i-1] not in seen:
            seen.add(A[i-1])
            cnt += 1
        L[i] = cnt
    # Compute R[i]: distinct count in suffix i..N
    seen.clear()
    R = [0]*(N+2)
    cnt = 0
    for i in range(N, 0, -1):
        if A[i-1] not in seen:
            seen.add(A[i-1])
            cnt += 1
        R[i] = cnt
    # Segment tree for dp2: range add, range max
    size = 1
    while size < N+2:
        size <<= 1
    tree = [0] * (2*size)
    lazy = [0] * (2*size)
    # build initial with L[1..N]
    for i in range(1, N+1):
        tree[size+i] = L[i]
    for i in range(size-1, 0, -1):
        tree[i] = max(tree[2*i], tree[2*i+1])
    def _apply(node, val):
        tree[node] += val
        lazy[node] += val
    def _push(node):
        v = lazy[node]
        if v:
            _apply(node*2, v)
            _apply(node*2+1, v)
            lazy[node] = 0
    def _update(node, nl, nr, l, r, val):
        if r < nl or nr < l:
            return
        if l <= nl and nr <= r:
            _apply(node, val)
            return
        _push(node)
        mid = (nl+nr)>>1
        _update(node*2, nl, mid, l, r, val)
        _update(node*2+1, mid+1, nr, l, r, val)
        tree[node] = max(tree[node*2], tree[node*2+1])
    def update(l, r, val):
        if l>r: return
        # clamp to [1..N]
        if l < 1: l = 1
        if r > N: r = N
        if l>r: return
        _update(1, 0, size-1, l, r, val)
    def _query(node, nl, nr, l, r):
        if r < nl or nr < l:
            return -10**18
        if l <= nl and nr <= r:
            return tree[node]
        _push(node)
        mid = (nl+nr)>>1
        return max(_query(node*2, nl, mid, l, r),
                   _query(node*2+1, mid+1, nr, l, r))
    def query(l, r):
        if l>r: return -10**18
        if l<1: l=1
        if r>N: r=N
        if l>r: return -10**18
        return _query(1, 0, size-1, l, r)
    # Compute dp2[j] for j=1..N
    last = [0]*(N+1)
    dp2 = [0]*(N+1)
    for j in range(1, N+1):
        x = A[j-1]
        p = last[x]
        # update range [p..j-1] by +1
        # if p==0, use 1..j-1
        if p==0:
            update(1, j-1, 1)
        else:
            update(p, j-1, 1)
        last[x] = j
        if j >= 2:
            dp2[j] = query(1, j-1)
        else:
            dp2[j] = 0
    # Final answer
    ans = 0
    # j is end of second segment, splits at j and j+1 so j in [2..N-1]
    for j in range(2, N):
        val = dp2[j] + R[j+1]
        if val > ans:
            ans = val
    print(ans)

if __name__ == "__main__":
    main()