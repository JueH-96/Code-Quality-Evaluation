class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                l = len(sub)
                min_changes = float('inf')
                for d in range(1, l):
                    if l % d == 0:
                        changes = 0
                        for start in range(d):
                            freq = {}
                            for idx in range(start, l, d):
                                if sub[idx] not in freq:
                                    freq[sub[idx]] = 0
                                freq[sub[idx]] += 1
                            max_freq = 0
                            for char in freq:
                                max_freq = max(max_freq, freq[char])
                            changes += (l // d - max_freq)
                        min_changes = min(min_changes, changes)
                if l > 0:
                    cost[i][j] = min_changes
                else:
                    cost[i][j] = 0

        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for x in range(i):
                    dp[i][j] = min(dp[i][j], dp[x][j - 1] + cost[x][i - 1])

        return dp[n][k]