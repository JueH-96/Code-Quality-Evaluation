class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        import bisect
        
        # Sort offers by their ending house
        offers.sort(key=lambda x: x[1])
        
        # Extract the end points into a separate list (for binary searching)
        ends = [offer[1] for offer in offers]
        
        # Prepare dp array, where dp[i] = max gold up to (and including) i-th offer
        dp = [0] * (len(offers) + 1)
        
        for i in range(len(offers)):
            start_i, end_i, gold_i = offers[i]
            
            # Find the rightmost offer j that ends strictly before start_i
            # (so that offers[i] does not overlap with offers[j])
            j = bisect.bisect_left(ends, start_i) - 1
            
            # Option 1: do not take the i-th offer
            not_take = dp[i]
            
            # Option 2: take the i-th offer + best we can do up to j
            take = gold_i
            if j >= 0:
                take += dp[j + 1]
                
            # dp[i+1] is the maximum of these two options
            dp[i+1] = max(not_take, take)
        
        return dp[len(offers)]