class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = [i for i in range(n) if s1[i] != s2[i]]
        
        if len(diff) % 2 != 0:
            return -1
        
        dp = [0] * (len(diff) + 1)
        dp[1] = x // 2
        
        for i in range(2, len(diff) + 1):
            dp[i] = min(dp[i - 1] + x // 2, dp[i - 2] + (diff[i - 1] - diff[i - 2]) // 2)
        
        return dp[-1] * 2