from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # A sufficiently small negative for infeasible states
        NEG_INF = -10**30
        
        # dp_prev[i] = max score choosing 1 element from b[0..i] for a[0]
        dp_prev = [0] * n
        for i in range(n):
            val = a[0] * b[i]
            if i == 0:
                dp_prev[i] = val
            else:
                dp_prev[i] = max(dp_prev[i-1], val)
        
        # Now build up for choosing 2nd, 3rd, 4th elements
        # s = 1 corresponds to a[1], ..., s = 3 corresponds to a[3]
        for s in range(1, 4):
            coeff = a[s]
            dp_curr = [NEG_INF] * n
            for i in range(n):
                if i == 0:
                    # Can't pick s+1 items ending at index 0 when s >= 1
                    dp_curr[i] = NEG_INF
                else:
                    # either skip b[i] or take it for a[s]
                    take = dp_prev[i-1] + coeff * b[i]
                    dp_curr[i] = max(dp_curr[i-1], take)
            dp_prev = dp_curr
        
        # The answer is the best score choosing all 4 at any ending index
        return dp_prev[-1]