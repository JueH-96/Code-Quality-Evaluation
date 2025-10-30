from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        N = 1 << n
        # dp[mask] = minimal time to break the set of locks indicated by bits in mask
        dp = [float('inf')] * N
        dp[0] = 0
        
        # Precompute popcounts if desired, or compute on the fly
        # Here we'll compute on the fly since n ≤ 8
        for mask in range(N):
            if dp[mask] == float('inf'):
                continue
            # current factor X after breaking popcount(mask) locks
            broken = mask.bit_count()
            X = 1 + broken * K
            # try breaking each lock j not yet broken
            for j in range(n):
                if not (mask & (1 << j)):
                    s = strength[j]
                    # time needed to accumulate at least s energy with factor X
                    t = (s + X - 1) // X
                    new_mask = mask | (1 << j)
                    dp[new_mask] = min(dp[new_mask], dp[mask] + t)
        
        return dp[N - 1]