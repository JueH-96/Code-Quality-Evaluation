class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [0] * (n + 1)
        j = 0
        for i in range(n):
            dp[i + 1] = dp[i]
            while j < len(offers) and offers[j][1] == i:
                dp[i + 1] = max(dp[i + 1], dp[offers[j][0]] + offers[j][2])
                j += 1
        return dp[n]