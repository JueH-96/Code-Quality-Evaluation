from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [float('-inf')] * n
        max_energy = float('-inf')
        
        for start in range(n):
            current_energy = 0
            for i in range(start, n, k):
                current_energy += energy[i]
                dp[i] = max(dp[i], current_energy)
                max_energy = max(max_energy, dp[i])
                if current_energy < 0:
                    current_energy = 0
        
        return max_energy