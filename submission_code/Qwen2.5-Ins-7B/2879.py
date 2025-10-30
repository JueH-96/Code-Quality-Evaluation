class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for parts in range(1, min(i, k) + 1):
                for j in range(i):
                    changes = 0
                    for d in range(1, (i - j) // 2 + 1):
                        if (i - j) % d == 0:
                            for m in range(d):
                                if s[j + m] != s[j + d + m]:
                                    changes += 1
                    dp[i][parts] = min(dp[i][parts], dp[j][parts - 1] + changes)
        
        return dp[n][k]