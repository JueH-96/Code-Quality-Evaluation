from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        # dp[mask] = minimum time to break the set of locks indicated by bits in mask
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        # Iterate over all subsets of locks
        for mask in range(1, 1 << n):
            # Try breaking each lock i that is in the current subset "mask" as the last one
            for i in range(n):
                if mask & (1 << i):
                    prev = mask ^ (1 << i)    # subset before breaking lock i
                    # current sword factor before breaking i
                    X_prev = 1 + (prev.bit_count()) * K
                    # minutes needed to accumulate enough energy for lock i
                    t = (strength[i] + X_prev - 1) // X_prev
                    dp[mask] = min(dp[mask], dp[prev] + t)
        
        # Answer is the time to break all locks (mask with all bits = 1)
        return dp[(1 << n) - 1]