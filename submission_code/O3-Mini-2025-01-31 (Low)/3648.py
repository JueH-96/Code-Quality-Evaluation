from typing import List
from functools import lru_cache

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # Check grid bounds helper. Returns True if (r,c) in [0, n)
        def in_grid(r, c):
            return 0 <= r < n and 0 <= c < n
        
        @lru_cache(maxsize=None)
        def dp(k, a, b, d):
            # k: moves taken so far (0-indexed). For child1 and child2, row = k.
            # a: for child1, number of (1,1) moves used so far; position child1 = (k, a)
            # b: for child2, net increment column delta (child2 position = (k, n-1+b))
            # d: for child3, net row difference added to starting row; position child3 = (n-1+d, k)
            # When k == n-1 all moves are done. Must check finishing conditions.
            if k == n - 1:
                # Final positions: child1: (n-1, a) must be (n-1, n-1) => a == n-1
                # child2: (n-1, n-1+b) must equal (n-1, n-1) => b == 0
                # child3: (n-1+d, n-1) must equal (n-1, n-1) => d == 0
                if a == n - 1 and b == 0 and d == 0:
                    # They all end in (n-1, n-1) but note that even though multiple children end there,
                    # the fruits are collected only once.
                    return fruits[n - 1][n - 1]
                else:
                    return float('-inf')
            
            # Current positions for the children
            r1, c1 = k, a
            r2, c2 = k, n - 1 + b
            r3, c3 = n - 1 + d, k
            
            # If any positions are out-of-bound, then this state is invalid.
            if not (in_grid(r1, c1) and in_grid(r2, c2) and in_grid(r3, c3)):
                return float('-inf')
            # Collect fruits from the current rooms (deduplicate overlapped cells)
            s = {}
            s[(r1, c1)] = fruits[r1][c1]
            s[(r2, c2)] = fruits[r2][c2]
            s[(r3, c3)] = fruits[r3][c3]
            current = sum(s.values())
            
            best = float('-inf')
            # Next move index is k+1
            nk = k + 1
            # child1: allowed: add0 => (1,0) ; add1 => (1,1)
            for na in (a, a + 1):
                # child1 next position = (nk, na)
                # Check bounds for child1:
                if not in_grid(nk, na):
                    continue
                # child2: delta from -1,0,1:
                for delta_b in (-1, 0, 1):
                    nb = b + delta_b
                    nc2 = n - 1 + nb
                    if not in_grid(nk, nc2):
                        continue
                    # child3: delta from -1,0,1:
                    for delta_d in (-1, 0, 1):
                        nd = d + delta_d
                        nr3 = n - 1 + nd
                        # child3 column is always nk.
                        if not in_grid(nr3, nk):
                            continue
                        val = dp(nk, na, nb, nd)
                        if val > best:
                            best = val
            return current + best
                        
        res = dp(0, 0, 0, 0)
        return res if res != float('-inf') else 0

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    fruits1 = [[1,2,3,4],[5,6,8,7],[9,10,11,12],[13,14,15,16]]
    print(sol.maxCollectedFruits(fruits1))  # Expected output: 100

    fruits2 = [[1,1],[1,1]]
    print(sol.maxCollectedFruits(fruits2))  # Expected output: 4