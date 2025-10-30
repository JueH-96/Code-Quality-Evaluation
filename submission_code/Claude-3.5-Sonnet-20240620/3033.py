class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_indices = [i for i in range(n) if s1[i] != s2[i]]
        m = len(diff_indices)
        
        if m % 2:
            return -1
        
        if m == 0:
            return 0
        
        dp = [0] * (m + 1)
        dp[1] = x
        
        for i in range(2, m + 1):
            dp[i] = min(dp[i-1] + x, dp[i-2] + 2 * min(x, diff_indices[i-1] - diff_indices[i-2]))
        
        return dp[m] // 2