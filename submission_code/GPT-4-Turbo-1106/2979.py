from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on end_i, then start_i, then gold_i in descending order
        offers.sort(key=lambda x: (x[1], x[0], -x[2]))
        
        # Initialize dp array where dp[i] represents the maximum gold we can get by selling houses up to house i
        dp = [0] * (n + 1)
        
        # Initialize pointer for offers
        offer_idx = 0
        
        # Iterate through each house
        for i in range(1, n + 1):
            # If no more offers or the current offer ends before the current house, we can't sell this house
            if offer_idx < len(offers) and offers[offer_idx][1] < i - 1:
                offer_idx += 1
            
            # Take the maximum of not selling the current house or selling it
            dp[i] = dp[i - 1]
            
            # Check if we can sell the current house with any offer
            while offer_idx < len(offers) and offers[offer_idx][1] == i - 1:
                start, end, gold = offers[offer_idx]
                dp[i] = max(dp[i], dp[start] + gold)
                offer_idx += 1
        
        # The last element in dp array will have the maximum gold we can achieve
        return dp[n]

# Example usage:
# sol = Solution()
# print(sol.maximizeTheProfit(5, [[0,0,1],[0,2,2],[1,3,2]]))  # Output: 3
# print(sol.maximizeTheProfit(5, [[0,0,1],[0,2,10],[1,3,2]]))  # Output: 10