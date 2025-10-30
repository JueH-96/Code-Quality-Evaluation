class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [0] * n
        for start, end, gold in offers:
            max_profit_before_start = 0
            if start > 0:
                max_profit_before_start = dp[start - 1]
            dp[end] = max(dp[end], max_profit_before_start + gold)
            if end > 0:
                dp[end] = max(dp[end], dp[end - 1])
        return dp[n - 1]