class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        sick_set = set(sick)
        dp = [0]*n
        dp[0] = 1
        if 1 not in sick_set:
            dp[1] = 2
        else:
            dp[1] = 1
        for i in range(2, n):
            if i not in sick_set:
                dp[i] = (dp[i-1] + dp[i-2]) % MOD
            else:
                dp[i] = dp[i-1]
        return dp[-1]