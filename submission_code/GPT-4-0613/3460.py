class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        requirements.sort()
        dp = [[0] * (n * n) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0]
            for j in range(1, i * (i - 1) // 2 + 1):
                dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % MOD
                if j >= i:
                    dp[i][j] = (dp[i][j] - dp[i - 1][j - i] + MOD) % MOD
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + dp[i - 1][i * (i - 1) // 2]
        res = dp[n][n * (n - 1) // 2]
        for end, cnt in requirements:
            if cnt > end * (end + 1) // 2:
                return 0
            res = res * (dp[end + 1][cnt] - prefix[cnt]) % MOD
        return res