from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # 1.  The first child’s path is forced to be the main diagonal.
        #     Collect these fruits once and afterwards treat the diagonal cells as empty.
        diag_sum = 0
        for i in range(n):
            diag_sum += fruits[i][i]
        
        NEG = -10**18                      # a value surely lower than any possible answer
        
        # ------------------------------------------------------------------
        # 2.  Child-2   (starts at (0 , n-1), one step per row, column ±1/0)
        # ------------------------------------------------------------------
        prev = [NEG] * n                   # dp for the previous row
        prev[n - 1] = fruits[0][n - 1]     # only the start cell is reachable in row 0
        
        for r in range(1, n):
            cur = [NEG] * n
            for c in range(n):
                best = prev[c]
                if c:            best = max(best, prev[c - 1])
                if c + 1 < n:    best = max(best, prev[c + 1])
                if best == NEG:  # this column cannot be reached
                    continue
                
                # diagonal cells are already empty
                val = 0 if r == c else fruits[r][c]
                cur[c] = best + val
            prev = cur
        gain_child2 = prev[n - 1]          # must finish at (n-1 , n-1)
        
        # ------------------------------------------------------------------
        # 3.  Child-3   (starts at (n-1 , 0), one step per column, row ±1/0)
        # ------------------------------------------------------------------
        prev = [NEG] * n                   # dp for the previous column
        prev[n - 1] = fruits[n - 1][0]     # only the start cell is reachable in column 0
        
        for c in range(1, n):
            cur = [NEG] * n
            for r in range(n):
                best = prev[r]
                if r:            best = max(best, prev[r - 1])
                if r + 1 < n:    best = max(best, prev[r + 1])
                if best == NEG:
                    continue
                
                val = 0 if r == c else fruits[r][c]
                cur[r] = best + val
            prev = cur
        gain_child3 = prev[n - 1]          # must finish at (n-1 , n-1)
        
        # ------------------------------------------------------------------
        # 4.  Total fruits collected
        # ------------------------------------------------------------------
        return diag_sum + gain_child2 + gain_child3