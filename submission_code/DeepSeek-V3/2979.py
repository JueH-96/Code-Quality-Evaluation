class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on end_i
        offers.sort(key=lambda x: x[1])
        
        # Initialize dp array
        dp = [0] * (n + 1)
        
        # Initialize a pointer for offers
        offer_ptr = 0
        
        for i in range(n):
            # Initialize dp[i+1] with dp[i] (not taking any offer for house i)
            dp[i+1] = dp[i]
            
            # Process all offers that end at i
            while offer_ptr < len(offers) and offers[offer_ptr][1] == i:
                start, end, gold = offers[offer_ptr]
                # Update dp[i+1] by considering the offer
                dp[i+1] = max(dp[i+1], dp[start] + gold)
                offer_ptr += 1
        
        return dp[n]