from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        """
        For every starting index i the total energy collected equals 
        energy[i] + energy[i+k] + energy[i+2k] + … (while the index is inside the array).
        Define dp[i] as this value.  
        Then              dp[i] = energy[i] + (dp[i+k] if i+k < n else 0)

        We can compute dp from right to left in O(n) and keep the maximum of all dp[i].
        """
        n = len(energy)
        dp = [0] * n
        best = -10**18          # smaller than the minimal possible sum

        for i in range(n - 1, -1, -1):
            dp[i] = energy[i]
            if i + k < n:                       # add the contribution that starts k steps ahead
                dp[i] += dp[i + k]
            best = max(best, dp[i])

        return best