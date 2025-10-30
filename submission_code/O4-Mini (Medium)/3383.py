from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] will hold the maximum energy starting from i and jumping by k
        dp = [0] * n
        max_energy = float('-inf')
        
        # Fill dp from the end towards the front
        for i in range(n - 1, -1, -1):
            if i + k < n:
                dp[i] = energy[i] + dp[i + k]
            else:
                dp[i] = energy[i]
            max_energy = max(max_energy, dp[i])
        
        return max_energy