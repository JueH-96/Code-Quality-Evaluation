from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        # Generate all pairs (i, j) with j - i >= 2 and record their product
        pairs = []
        for i in range(n - 2):
            vi = nums[i]
            for j in range(i + 2, n):
                pairs.append((vi * nums[j], i, j))
        # No pairs => no quadruples
        if not pairs:
            return 0
        # Sort by product, then by start index ascending
        pairs.sort()
        m = len(pairs)
        ans = 0

        # Fenwick (BIT) with stamping to avoid clearing
        class FenwickStamped:
            __slots__ = ('n', 'f', 'ver', 'curv')
            def __init__(self, n):
                self.n = n
                self.f = [0] * (n + 1)
                self.ver = [0] * (n + 1)
                self.curv = 0
            def new_version(self):
                self.curv += 1
            def update(self, idx: int, delta: int):
                v = self.curv
                f = self.f
                ver = self.ver
                n = self.n
                i = idx
                while i <= n:
                    if ver[i] != v:
                        ver[i] = v
                        f[i] = delta
                    else:
                        f[i] += delta
                    i += i & -i
            def query(self, idx: int) -> int:
                res = 0
                v = self.curv
                f = self.f
                ver = self.ver
                i = idx
                while i > 0:
                    if ver[i] == v:
                        res += f[i]
                    i -= i & -i
                return res

        bit = FenwickStamped(n)
        idx0 = 0
        # Scan through sorted pairs by product, group equal-products
        while idx0 < m:
            prod = pairs[idx0][0]
            start = idx0
            idx1 = idx0 + 1
            while idx1 < m and pairs[idx1][0] == prod:
                idx1 += 1
            end = idx1
            group_size = end - start
            # Only groups with at least 2 pairs can form a quadruple
            if group_size > 1:
                bit.new_version()
                add_ptr = start
                # pairs[start:end] are sorted by start-index ascending
                for k in range(start, end):
                    _, xj, yj = pairs[k]
                    # we only bring into BIT those i with xi <= xj - 2
                    thresh = xj - 2
                    while add_ptr < end and pairs[add_ptr][1] <= thresh:
                        _, xi, yi = pairs[add_ptr]
                        # store yi in BIT at position yi+1
                        bit.update(yi + 1, 1)
                        add_ptr += 1
                    # now BIT holds all i with xi <= xj-2
                    # we need yi in [ xj+2 .. yj-2 ]
                    low = xj + 2
                    high = yj - 2
                    if low <= high:
                        # clamp to valid range
                        if low < 0: low = 0
                        if high >= n: high = n - 1
                        lidx = low + 1
                        ridx = high + 1
                        if lidx <= ridx:
                            ans += bit.query(ridx) - bit.query(lidx - 1)
            idx0 = end

        return ans