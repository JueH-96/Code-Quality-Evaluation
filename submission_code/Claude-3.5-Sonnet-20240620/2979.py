class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        dp = [0] * (n + 1)
        offers.sort(key=lambda x: x[1])
        
        offer_index = 0
        for house in range(n):
            dp[house + 1] = dp[house]
            
            while offer_index < len(offers) and offers[offer_index][1] == house:
                start, end, gold = offers[offer_index]
                dp[house + 1] = max(dp[house + 1], dp[start] + gold)
                offer_index += 1
        
        return dp[n]