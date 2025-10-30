class Solution:
    def minimumCost(self, source: str, target: str, original: list[str], changed: list[str], cost: list[int]) -> int:
        n = len(source)
        m = len(original)
        dp = [[float('inf')] * 26 for _ in range(n + 1)]
        dp[0][ord(source[0]) - ord('a')] = 0
        for i in range(1, n + 1):
            for j in range(26):
                dp[i][j] = min(dp[i][j], dp[i - 1][j])
                for k in range(m):
                    if original[k] == chr(j + ord('a')):
                        dp[i][ord(changed[k]) - ord('a')] = min(dp[i][ord(changed[k]) - ord('a')], dp[i - 1][j] + cost[k])
        res = min(dp[n][ord(target[i]) - ord('a')] for i in range(n))
        return -1 if res == float('inf') else res