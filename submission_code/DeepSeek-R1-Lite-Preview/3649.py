from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        full_mask = (1 << n) - 1
        INF = float('inf')
        dp = [INF] * (1 << n)
        dp[0] = 0  # No locks broken, time = 0

        # Precompute the number of set bits for each mask
        set_bits = [bin(mask).count('1') for mask in range(1 << n)]

        for mask in range(1 << n):
            if dp[mask] == INF:
                continue
            X = 1 + K * set_bits[mask]
            for i in range(n):
                if not (mask & (1 << i)):
                    time_needed = (strength[i] + X - 1) // X  # ceil(strength[i] / X)
                    new_mask = mask | (1 << i)
                    dp[new_mask] = min(dp[new_mask], dp[mask] + time_needed)
        
        return dp[full_mask]