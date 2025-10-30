class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        low, high = int(low), int(high)
        dp = [[0]*10 for _ in range(101)]
        for i in range(1, 10):
            dp[1][i] = 1
        for i in range(2, 101):
            for j in range(10):
                if j > 0:
                    dp[i][j] += dp[i-1][j-1]
                if j < 9:
                    dp[i][j] += dp[i-1][j+1]
                dp[i][j] %= MOD
        def count(n):
            if n == 0:
                return 1
            s = str(n)
            l = len(s)
            res = sum(sum(dp[i]) for i in range(1, l))
            prev = int(s[0])
            for i in range(1, l):
                curr = int(s[i])
                for j in range(prev):
                    res += dp[l-i][j]
                if abs(prev-curr) != 1:
                    break
                if i == l-1:
                    res += 1
                prev = curr
            return res % MOD
        return (count(high) - count(low-1) + MOD) % MOD