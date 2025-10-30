class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        dp = [[[0]*k for _ in range(2)] for _ in range(n)]
        dp[0][0][0] = 9
        for i in range(1, k):
            dp[0][0][i] = 1
        for i in range(1, n):
            for j in range(k):
                for d in range(10):
                    for e in range(10):
                        for f in range(2):
                            if f == 1 and d > e:
                                break
                            if d != e:
                                dp[i][f or d>0][(j*10+d)%k] += dp[i-1][f][j]
                                dp[i][f or d>0][(j*10+d)%k] %= MOD
                            else:
                                dp[i][f or d>0][(j*10+d)%k] += dp[i-1][f][(j*10+d)%k]
                                dp[i][f or d>0][(j*10+d)%k] %= MOD
        return sum(dp[-1][1])%MOD