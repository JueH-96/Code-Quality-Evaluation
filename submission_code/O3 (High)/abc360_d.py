import sys
import bisect

# ---------- Fenwick Tree ----------
class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int = 1):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix_sum(self, i: int) -> int:
        res = 0
        while i:
            res += self.bit[i]
            i -= i & -i
        return res

    def range_sum(self, l: int, r: int) -> int:          # 1-based, inclusive
        if l > r:
            return 0
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

# ---------- Main ----------
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    it = iter(input_data)
    N = int(next(it))
    T = int(next(it))
    S = list(next(it).strip())
    X = [int(next(it)) for _ in range(N)]

    # Coordinate compression
    coords = sorted(X)
    M = len(coords)
    # Two Fenwick trees:
    #   right_fw  : ants that have appeared so far whose direction = 1  (move right)
    #   left_fw   : ants that have appeared so far whose direction = 0  (move left)
    right_fw = Fenwick(M)
    left_fw  = Fenwick(M)

    TWO_T = 2 * T
    ans = 0

    for idx, (x, dir_char) in enumerate(zip(X, S)):
        # Coordinate indices (1-based for Fenwick)
        # queries need bisect on coords which is already sorted list of unique coords
        if dir_char == '0':                       # current ant moves left
            l = x - TWO_T
            r = x - 1
            left_idx = bisect.bisect_left(coords, l) + 1       # first >= l
            right_idx = bisect.bisect_right(coords, r)         # count of <= r (already 1-based)
            if left_idx <= right_idx:
                ans += right_fw.range_sum(left_idx, right_idx)
        else:                                     # current ant moves right
            l = x + 1
            r = x + TWO_T
            left_idx = bisect.bisect_left(coords, l) + 1
            right_idx = bisect.bisect_right(coords, r)
            if left_idx <= right_idx:
                ans += left_fw.range_sum(left_idx, right_idx)

        # Insert current ant into its corresponding tree
        comp_idx = bisect.bisect_left(coords, x) + 1
        if dir_char == '1':
            right_fw.add(comp_idx, 1)
        else:
            left_fw.add(comp_idx, 1)

    print(ans)

if __name__ == "__main__":
    main()