from collections import defaultdict

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        end_house_offers = defaultdict(list)
        for start, end, gold in offers:
            end_house_offers[end].append((start, gold))
        
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i+1] = dp[i]
            for start, gold in end_house_offers[i]:
                profit_with_offer = gold + dp[start]
                dp[i+1] = max(dp[i+1], profit_with_offer)
                
        return dp[n]