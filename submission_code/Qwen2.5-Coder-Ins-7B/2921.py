class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count_steps(n):
            if n == 0:
                return 1
            if n == 1:
                return 10
            dp = [0] * 10
            dp[0] = 1
            dp[1] = 2
            for i in range(2, n):
                dp[i] = (dp[i-1] + dp[i-2]) % MOD
            return dp[n-1]
        
        def count_numbers(num):
            num = str(num)
            n = len(num)
            dp = [[0] * 10 for _ in range(n)]
            dp[0] = [1] * 10
            for i in range(1, n):
                for j in range(10):
                    if j > 0:
                        dp[i][j] += dp[i-1][j-1]
                    if j < 9:
                        dp[i][j] += dp[i-1][j+1]
                    dp[i][j] %= MOD
            count = 0
            for j in range(10):
                if str(j) <= num[0]:
                    count += dp[n-1][j]
                    if str(j) == num[0]:
                        count -= dp[n-2][j]
                    count %= MOD
            return count
        
        return (count_numbers(int(high)) - count_numbers(int(low) - 1) + count_steps(len(high)) - count_steps(len(low) - 1)) % MOD