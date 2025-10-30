from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] will store the total energy obtainable starting at index i.
        dp = [0] * n
        dp[-1] = energy[-1]
        max_energy = dp[-1]
        
        # Process from the second-last element backward to index 0.
        for i in range(n - 2, -1, -1):
            # If a jump from index i is valid, add the dp value from the jumped index.
            if i + k < n:
                dp[i] = energy[i] + dp[i + k]
            else:
                dp[i] = energy[i]
            max_energy = max(max_energy, dp[i])
        
        return max_energy