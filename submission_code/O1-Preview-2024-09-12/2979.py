class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers_at_end = [[] for _ in range(n)]
        for start, end, gold in offers:
            offers_at_end[end].append((start, gold))
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for start, gold in offers_at_end[i]:
                dp[i + 1] = max(dp[i + 1], dp[start] + gold)
        return dp[n]