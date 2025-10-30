class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        
        # We'll fill dp from the end, where dp[i] = energy[i] + dp[i+k] if i+k is still within range.
        for i in range(n - 1, -1, -1):
            dp[i] = energy[i]
            if i + k < n:
                dp[i] += dp[i + k]
                
        # The answer is the maximum dp value for any starting position.
        return max(dp)