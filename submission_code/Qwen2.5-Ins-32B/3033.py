class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diff_indices = [i for i in range(len(s1)) if s1[i] != s2[i]]
        n = len(diff_indices)
        
        if n % 2 != 0:
            return -1
        
        if n == 0:
            return 0
        
        dp = [0] * (n + 1)
        dp[1] = x
        
        for i in range(2, n + 1):
            cost = diff_indices[i - 1] - diff_indices[i - 2]
            dp[i] = min(dp[i - 1] + cost, dp[i - 2] + x)
        
        return dp[n]