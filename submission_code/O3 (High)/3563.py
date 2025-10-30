from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # number of rows ( ≤ 10 )
        m = len(grid)
        
        # collect for every value the set of rows in which it appears
        value_to_rows = {}
        for r, row in enumerate(grid):
            for v in row:
                value_to_rows.setdefault(v, set()).add(r)
        
        # bit-mask DP – mask represents which rows are already occupied
        FULL = 1 << m
        dp = [-float('inf')] * FULL
        dp[0] = 0
        
        for val, rows in value_to_rows.items():
            # copy current table so that we never use the same value twice
            ndp = dp.copy()
            for mask in range(FULL):
                if dp[mask] < 0:          # unreachable state
                    continue
                # try assigning the present value to every possible row
                for r in rows:
                    if mask & (1 << r):   # row already taken
                        continue
                    new_mask = mask | (1 << r)
                    ndp[new_mask] = max(ndp[new_mask], dp[mask] + val)
            dp = ndp
        
        return max(dp)