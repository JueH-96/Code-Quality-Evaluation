class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        dp = [0] * (n + 1)
        offers.sort(key=lambda x: x[1])
        
        for i in range(len(offers)):
            start, end, gold = offers[i]
            dp[end + 1] = max(dp[end + 1], dp[start] + gold)
            dp[end + 1] = max(dp[end + 1], dp[end])
        
        return max(dp)