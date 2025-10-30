class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers by end position
        offers.sort(key=lambda x: x[1])
        
        # dp[i] represents max profit possible up to house i
        dp = [0] * n
        
        curr_offer = 0
        for house in range(n):
            # Copy previous max profit
            if house > 0:
                dp[house] = dp[house-1]
            
            # Process all offers ending at current house
            while curr_offer < len(offers) and offers[curr_offer][1] == house:
                start = offers[curr_offer][0]
                gold = offers[curr_offer][2]
                
                # Max profit = max(current profit, profit before start + current offer)
                prev_profit = 0 if start == 0 else dp[start-1]
                dp[house] = max(dp[house], prev_profit + gold)
                
                curr_offer += 1
                
        return dp[n-1]