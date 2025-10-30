class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] will store the total energy gained by starting at i and jumping by k
        dp = [0] * n
        max_energy = float('-inf')
        
        # fill dp from the back so dp[i+k] is already known when computing dp[i]
        for i in range(n - 1, -1, -1):
            dp[i] = energy[i] + (dp[i + k] if i + k < n else 0)
            max_energy = max(max_energy, dp[i])
        
        return max_energy