from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] will hold the maximum energy starting from i
        dp = [0] * n
        max_energy = float('-inf')
        
        # Fill dp from the end towards the front
        for i in range(n - 1, -1, -1):
            # If we can jump from i to i+k, include that dp value
            if i + k < n:
                dp[i] = energy[i] + dp[i + k]
            else:
                dp[i] = energy[i]
            # Track the maximum
            if dp[i] > max_energy:
                max_energy = dp[i]
        
        return max_energy