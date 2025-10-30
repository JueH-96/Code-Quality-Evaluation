from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Prefix sums: S[i] = sum of nums[0:i]
        S = [0] * (n + 1)
        for i in range(n):
            S[i + 1] = S[i] + nums[i]
        
        # Sentinel for -infinity
        NEG_INF = -10**30
        
        # DP rolling arrays: prev[j] = DP for j-1 segments, curr = for j segments
        # DP[j][i] = max strength using j segments in first i elements.
        # We store arrays of length n+1 for positions 0..n.
        prev = [0] * (n + 1)  # DP[0][*] = 0
        
        # For each segment count j = 1..k
        for j in range(1, k + 1):
            # Coefficient for the j-th subarray
            coeff = ((-1) ** (j + 1)) * (k - j + 1)
            curr = [NEG_INF] * (n + 1)
            
            # DP[j][0] = -inf (cannot pick j>0 segments from 0 elements)
            best = prev[0] - coeff * S[0]  # will be NEG_INF for j>1, zero for j=1
            
            for i in range(1, n + 1):
                # Option 1: do not end a segment at i -> inherit DP[j][i-1]
                v = curr[i - 1]
                # Option 2: end j-th segment at i -> best + coeff * S[i]
                # where best = max over p<i of (DP[j-1][p] - coeff * S[p])
                cand = best + coeff * S[i]
                if cand > v:
                    v = cand
                curr[i] = v
                
                # Update best for future i: consider p = i
                val = prev[i] - coeff * S[i]
                if val > best:
                    best = val
            
            # Roll arrays: now prev becomes curr for next j
            prev = curr
        
        # The answer is DP[k][n]
        return prev[n]