class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        if len(s1) + len(s2) != len(s3):
            return -1

        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) - 1, -1, -1):
            for j in range(len(s2) - 1, -1, -1):
                k = len(s1) + len(s2) - i - j
                if s1[i] == s3[k] and s2[j] != s3[k]:
                    dp[i][j] = dp[i+1][j] + 1
                elif s1[i] != s3[k] and s2[j] == s3[k]:
                    dp[i][j] = dp[i][j+1] + 1
                elif s1[i] == s3[k] and s2[j] == s3[k]:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1])
                else:
                    dp[i][j] = float('inf')

        return dp[0][0] if dp[0][0] != float('inf') else -1