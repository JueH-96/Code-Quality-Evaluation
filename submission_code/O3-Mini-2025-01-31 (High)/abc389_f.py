def main():
    import sys,sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    contests = [ (int(next(it)), int(next(it))) for _ in range(N) ]
    Q = int(next(it))
    qs = [ int(next(it)) for _ in range(Q) ]
    # We only need to simulate dp for initial ratings 1 .. n
    n = max(qs)
    st = SegTree(n)
    # Process contests in order.
    # For each contest (L,R) – update dp[x] (for those x whose current dp[x] is in [L,R]) by +1.
    for L, R in contests:
        lo = st.lower_bound(L)         # first index (0-indexed) with dp >= L
        hi = st.lower_bound(R + 1)       # first index with dp >= R+1, so indices in [lo, hi-1] have value in [L,R]
        if lo < hi:
            st.update(lo, hi - 1, 1)
    # Now answer queries: for an initial rating X, the answer is dp[X-1]
    out_lines = []
    for x in qs:
        # Query the point x-1 (0-indexed)
        out_lines.append(str(st.query(x - 1)))
    sys.stdout.write("
".join(out_lines))

# --- Segment Tree with lazy propagation supporting range-add, point query, and lower_bound ---
class SegTree:
    __slots__ = 'n','size','tree','lazy'
    def __init__(self, n):
        # Build an array dp[0…n–1] with dp[i] = i+1 (i.e. initial rating = index+1)
        self.n = n
        size = 1
        while size < n:
            size *= 2
        self.size = size
        # tree is 1-indexed; leaves appear at indices [size, size+n)
        self.tree = [0] * (2 * size)
        self.lazy = [0] * (2 * size)
        for i in range(n):
            self.tree[size + i] = i + 1
        for i in range(n, size):
            self.tree[size + i] = float("inf")
        for i in range(size - 1, 0, -1):
            # Internal node value is the min of its two children.
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])
    
    # Push down lazy value from node i to its children.
    def _push(self, i):
        if self.lazy[i]:
            self._apply(2 * i, self.lazy[i])
            self._apply(2 * i + 1, self.lazy[i])
            self.lazy[i] = 0
    
    # Apply an update “val” to node i.
    def _apply(self, i, val):
        self.tree[i] += val
        self.lazy[i] += val

    # Recursively update the range [l, r] (0-indexed) – here lo/hi are the node’s interval.
    def _update(self, i, lo, hi, l, r, val):
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            self.tree[i] += val
            self.lazy[i] += val
        else:
            self._push(i)
            mid = (lo + hi) // 2
            self._update(2 * i, lo, mid, l, r, val)
            self._update(2 * i + 1, mid + 1, hi, l, r, val)
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])
    
    # Public update: add val to dp[x] for all x in [l, r] (0-indexed).
    def update(self, l, r, val):
        self._update(1, 0, self.size - 1, l, r, val)
    
    # Query the value at index idx (0-indexed).
    def _query(self, i, lo, hi, idx):
        if lo == hi:
            return self.tree[i]
        self._push(i)
        mid = (lo + hi) // 2
        if idx <= mid:
            return self._query(2 * i, lo, mid, idx)
        else:
            return self._query(2 * i + 1, mid + 1, hi, idx)
    def query(self, idx):
        return self._query(1, 0, self.size - 1, idx)
    
    # Find the smallest index x (0-indexed) such that dp[x] >= X.
    # (If no such index exists we return self.n.)
    def lower_bound(self, X):
        # Because dp is strictly increasing, we can check boundaries.
        # If the smallest dp value [at index 0] is already >= X then answer is 0.
        val0 = self.query(0)
        if val0 >= X:
            return 0
        # If dp[n-1] < X then no answer in [0, n-1]
        val_last = self.query(self.n - 1)
        if val_last < X:
            return self.n
        i = 1
        lo = 0
        hi = self.size - 1
        # Descend the tree: always push lazy values on the way.
        while lo < hi:
            self._push(i)
            mid = (lo + hi) // 2
            # Check if the minimum in the left child is at least X.
            if self.tree[2 * i] >= X:
                i = 2 * i
                hi = mid
            else:
                i = 2 * i + 1
                lo = mid + 1
        return lo

# Do not forget to call main()!
if __name__ == '__main__':
    main()