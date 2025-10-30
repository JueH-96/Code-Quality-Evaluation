class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        def is_semi_palindrome(sub: str) -> int:
            n = len(sub)
            min_changes = float('inf')
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for i in range(d):
                        group = sub[i::d]
                        m = len(group)
                        for j in range(m // 2):
                            if group[j] != group[m - j - 1]:
                                changes += 1
                    min_changes = min(min_changes, changes)
            return min_changes

        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(j, i + 1):
                    dp[i][j] = min(dp[i][j], dp[l - 1][j - 1] + is_semi_palindrome(s[l - 1:i]))

        return dp[n][k]