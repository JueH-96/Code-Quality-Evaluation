class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        dp = [0] * (n + 1)
        offers.sort(key=lambda x: x[1])  # Sort offers by end index

        for start, end, gold in offers:
            # We can only consider selling to this buyer if we don't overlap with previous sales
            dp[end + 1] = max(dp[end + 1], dp[start] + gold)

        # Fill dp array to ensure we carry forward the maximum profit
        for i in range(1, n + 1):
            dp[i] = max(dp[i], dp[i - 1])

        return dp[n]