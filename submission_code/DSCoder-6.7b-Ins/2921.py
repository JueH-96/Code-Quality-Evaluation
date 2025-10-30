class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        dp = [0]*10
        dp[0] = 1
        for i in range(1, 10):
            dp[i] = dp[i-1]*2
            dp[i] %= MOD
        
        n1, n2 = len(low), len(high)
        dp2 = [0]*(n2+1)
        dp2[1] = 10
        for i in range(2, n2+1):
            dp2[i] = dp2[i-1]*10
            dp2[i] %= MOD
        
        res = dp2[n2]
        if n1 > 1:
            res -= dp2[n1-1]
            res %= MOD
        
        a, b = 0, 0
        for i in range(n1-1, -1, -1):
            a = a*10 + int(low[i])
            b = b*10 + int(high[i])
            res += b - a + 1
            res %= MOD
        
        return res