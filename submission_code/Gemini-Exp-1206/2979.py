from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [0] * (n + 1)
        offer_index = 0
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            while offer_index < len(offers) and offers[offer_index][1] == i - 1:
                start, end, gold = offers[offer_index]
                dp[i] = max(dp[i], dp[start] + gold)
                offer_index += 1
        return dp[n]