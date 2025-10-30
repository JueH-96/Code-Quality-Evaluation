import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        if not offers:
            return 0
        
        # Sort offers by their end points
        sorted_offers = sorted(offers, key=lambda x: x[1])
        m = len(sorted_offers)
        ends = [x[1] for x in sorted_offers]
        dp = [0] * m
        
        # Initialize DP for the first offer
        dp[0] = sorted_offers[0][2]
        
        for i in range(1, m):
            start = sorted_offers[i][0]
            target = start - 1
            # Find the latest offer that ends before or at target
            j = bisect.bisect_right(ends, target, 0, i) - 1
            if j >= 0:
                take = sorted_offers[i][2] + dp[j]
            else:
                take = sorted_offers[i][2]
            not_take = dp[i-1]
            dp[i] = max(take, not_take)
        
        return dp[-1]