from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Union-Find (Disjoint Set Union) with path compression & union by rank
        class DSU:
            def __init__(self, n):
                self.p = list(range(n))
                self.r = [0]*n
            def find(self, x):
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False
                # union by rank
                if self.r[rx] < self.r[ry]:
                    rx, ry = ry, rx
                self.p[ry] = rx
                if self.r[rx] == self.r[ry]:
                    self.r[rx] += 1
                return True

        # Separate nums into those <= threshold (candidates for edges)
        # and those > threshold (always isolated)
        small = []
        large_count = 0
        for v in nums:
            if v <= threshold:
                small.append(v)
            else:
                large_count += 1

        # If no small numbers, all are isolated
        if not small:
            return large_count

        # Map each small value to an index in DSU
        small.sort()
        idx = {v:i for i,v in enumerate(small)}
        m = len(small)
        dsu = DSU(m)

        # rep[L] = the first small divisor we saw for multiple L
        # once we see another small divisor for same L, we union them
        rep = [ -1 ] * (threshold + 1)

        # For each small value d, iterate its multiples L = d,2d,... <= threshold
        # and union d with whichever small value was first seen at rep[L]
        for d in small:
            di = idx[d]
            # iterate multiples
            step = d
            for L in range(d, threshold+1, step):
                j = rep[L]
                if j == -1:
                    rep[L] = di
                else:
                    # union the two small values
                    dsu.union(di, j)

        # Count distinct roots among small
        roots = set()
        for i in range(m):
            roots.add(dsu.find(i))

        # total components = components among small + each large isolated
        return len(roots) + large_count