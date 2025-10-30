import sys

# -------- Segment Tree supporting range add & range max --------------- #
class SegTree:
    def __init__(self, base):
        """
        base : 1-indexed list/array with length n+1 (base[0] is dummy)
        the tree keeps values equal to base[i]
        """
        self.n = len(base) - 1                      # number of real elements
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.mx   = [0] * (self.size * 2)           # maximum values
        self.lazy = [0] * (self.size * 2)           # pending additions

        # build leaves
        for i in range(1, self.n + 1):
            self.mx[self.size + i - 1] = base[i]
        # build internal nodes
        for i in range(self.size - 1, 0, -1):
            self.mx[i] = max(self.mx[i << 1], self.mx[i << 1 | 1])

    def _apply(self, idx, val):
        self.mx[idx]   += val
        self.lazy[idx] += val

    def _push(self, idx):
        if self.lazy[idx]:
            self._apply(idx << 1,     self.lazy[idx])
            self._apply(idx << 1 | 1, self.lazy[idx])
            self.lazy[idx] = 0

    # add `val` to every position in [l, r]   (1-indexed, inclusive)
    def range_add(self, l, r, val):
        if l > r:
            return
        self._range_add(1, 1, self.size, l, r, val)

    def _range_add(self, idx, tl, tr, l, r, val):
        if l == tl and r == tr:
            self._apply(idx, val)
            return
        self._push(idx)
        tm = (tl + tr) >> 1
        if r <= tm:
            self._range_add(idx << 1, tl, tm, l, r, val)
        elif l > tm:
            self._range_add(idx << 1 | 1, tm + 1, tr, l, r, val)
        else:
            self._range_add(idx << 1,     tl, tm, l, tm, val)
            self._range_add(idx << 1 | 1, tm + 1, tr, tm + 1, r, val)
        self.mx[idx] = max(self.mx[idx << 1], self.mx[idx << 1 | 1])

    # maximum on interval [l, r]  (1-indexed, inclusive)
    def range_max(self, l, r):
        if l > r:
            return -10**18
        return self._range_max(1, 1, self.size, l, r)

    def _range_max(self, idx, tl, tr, l, r):
        if l == tl and r == tr:
            return self.mx[idx]
        self._push(idx)
        tm = (tl + tr) >> 1
        if r <= tm:
            return self._range_max(idx << 1, tl, tm, l, r)
        elif l > tm:
            return self._range_max(idx << 1 | 1, tm + 1, tr, l, r)
        else:
            return max(
                self._range_max(idx << 1,     tl, tm, l, tm),
                self._range_max(idx << 1 | 1, tm + 1, tr, tm + 1, r)
            )

# ----------------------------- Main ----------------------------------- #
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.read().strip().split()
    if not input_data:
        return
    N   = int(input_data[0])
    A   = [0] + list(map(int, input_data[1:]))

    # prefix distinct counts f[i]
    seen = [False] * (N + 1)
    f    = [0] * (N + 1)
    cnt  = 0
    for i in range(1, N + 1):
        x = A[i]
        if not seen[x]:
            seen[x] = True
            cnt    += 1
        f[i] = cnt

    # suffix distinct counts s[i] : distinct elements in A[i..N]
    seen = [False] * (N + 1)
    s    = [0] * (N + 2)          # s[N+1] stays 0
    cnt  = 0
    for i in range(N, 0, -1):
        x = A[i]
        if not seen[x]:
            seen[x] = True
            cnt    += 1
        s[i] = cnt

    # segment tree initialised with f[i] ( because d_i = 0 at j = 1 )
    seg = SegTree(f)

    # last occurrence of each value (0 if not seen yet)
    last_pos = [0] * (N + 1)
    last_pos[A[1]] = 1            # we have processed position 1 before loop

    ans = 0
    # iterate possible position j (end of the middle segment)
    for j in range(2, N):         # j runs from 2 .. N-1
        val  = A[j]
        prev = last_pos[val]      # last occurrence before j (0 if none)
        last_pos[val] = j

        L = max(prev, 1)
        R = j - 1
        if L <= R:
            seg.range_add(L, R, 1)  # new value appears for all i in [L, R]

        best_left_middle = seg.range_max(1, R)
        candidate        = best_left_middle + s[j + 1]
        if candidate > ans:
            ans = candidate

    print(ans)


if __name__ == "__main__":
    main()