class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        arr = []
        k = 1
        while True:
            val = k ** x
            if val > n:
                break
            arr.append(val)
            k += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for num in arr:
            for j in range(n, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        return dp[n]