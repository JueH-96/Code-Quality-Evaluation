import sys

# -------------------- Segment Tree -------------------- #
class SegTree:
    """
    segment tree that supports
        1. range add (increment every element in [l, r] by v)
        2. range max query
    """
    __slots__ = ("n", "size", "data", "lazy")

    def __init__(self, n: int):
        self.n = n
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        self.data = [0] * (size << 1)
        self.lazy = [0] * (size << 1)

    # apply increment to a node
    def _apply(self, node: int, val: int):
        self.data[node] += val
        self.lazy[node] += val

    # push lazy value to children
    def _push(self, node: int):
        if self.lazy[node]:
            val = self.lazy[node]
            self._apply(node << 1, val)
            self._apply(node << 1 | 1, val)
            self.lazy[node] = 0

    # internal range add
    def _range_add(self, l: int, r: int, val: int, node: int, nl: int, nr: int):
        if r < nl or nr < l:
            return
        if l <= nl and nr <= r:
            self._apply(node, val)
            return
        self._push(node)
        mid = (nl + nr) >> 1
        self._range_add(l, r, val, node << 1, nl, mid)
        self._range_add(l, r, val, node << 1 | 1, mid + 1, nr)
        self.data[node] = max(self.data[node << 1], self.data[node << 1 | 1])

    # public range add (inclusive)
    def range_add(self, l: int, r: int, val: int = 1):
        self._range_add(l, r, val, 1, 0, self.size - 1)

    # internal range max
    def _range_max(self, l: int, r: int, node: int, nl: int, nr: int) -> int:
        if r < nl or nr < l:
            return -1  # since counts are non-negative
        if l <= nl and nr <= r:
            return self.data[node]
        self._push(node)
        mid = (nl + nr) >> 1
        left = self._range_max(l, r, node << 1, nl, mid)
        right = self._range_max(l, r, node << 1 | 1, mid + 1, nr)
        return left if left >= right else right

    # public range max (inclusive)
    def range_max(self, l: int, r: int) -> int:
        return self._range_max(l, r, 1, 0, self.size - 1)

# -------------------- Main -------------------- #
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W, N = map(int, input().split())

    bars = []  # (row, left, length, index)
    for idx in range(N):
        R, C, L = map(int, input().split())
        bars.append((R, C, L, idx))

    # process bars from bottom to top (larger R first)
    bars.sort(key=lambda x: (-x[0], x[3]))

    seg = SegTree(W)              # column indices are 0-based in the tree
    answer = [0] * N

    for R, C, L, idx in bars:
        l = C - 1                 # 0-based
        r = C + L - 2             # inclusive
        cnt_max = seg.range_max(l, r)     # bars already stacked below
        final_row = H - cnt_max            # formula explained in analysis
        answer[idx] = final_row
        seg.range_add(l, r, 1)            # add current bar to every column it spans

    # output
    out_lines = "
".join(map(str, answer))
    sys.stdout.write(out_lines + "
")


if __name__ == "__main__":
    main()