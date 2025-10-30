class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        if x < 1:
            return -1
        n = len(s1)
        mismatches = [i for i in range(n) if s1[i] != s2[i]]
        m = len(mismatches)
        if m == 0:
            return 0
        if m % 2 == 1:
            return -1
        dp = [[float('inf')] * m for _ in range(m)]
        for i in range(m):
            dp[i][i] = 0
        for i in range(m - 1):
            if mismatches[i + 1] - mismatches[i] == 1:
                dp[i][i + 1] = 1
        for length in range(2, m):
            for start in range(m - length):
                end = start + length
                for mid in range(start + 1, end, 2):
                    dp[start][end] = min(dp[start][end], dp[start][mid] + dp[mid][end])
                if (end - start) % 2 == 0:
                    dp[start][end] = min(dp[start][end], x * ((end - start) // 2))
        return dp[0][m - 1]