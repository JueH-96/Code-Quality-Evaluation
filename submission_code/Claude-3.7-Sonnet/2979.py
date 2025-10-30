class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Group offers by their end index
        end_to_offers = {}
        for start, end, gold in offers:
            if end not in end_to_offers:
                end_to_offers[end] = []
            end_to_offers[end].append((start, gold))
        
        # dp[i] represents the maximum gold we can earn for houses 0 to i
        dp = [0] * n
        
        for i in range(n):
            # Default case: carry over the maximum profit from the previous house
            dp[i] = dp[i-1] if i > 0 else 0
            
            # Consider all offers that end at house i
            if i in end_to_offers:
                for start, gold in end_to_offers[i]:
                    # If we include this offer, we add its gold to the maximum gold
                    # we can earn from houses before the offer's start
                    prev_max = dp[start-1] if start > 0 else 0
                    dp[i] = max(dp[i], prev_max + gold)
        
        return dp[n-1]