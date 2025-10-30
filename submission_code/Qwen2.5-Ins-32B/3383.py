from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        dp[-1] = energy[-1]
        for i in range(n - 2, -1, -1):
            dp[i] = energy[i] + (dp[i + k] if i + k < n else 0)
        return max(dp)