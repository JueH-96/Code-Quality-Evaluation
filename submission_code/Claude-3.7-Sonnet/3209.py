class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        memo = {}
        
        def dp(i, free):
            # Base case: we've processed all fruits
            if i > n:
                return 0
            
            # Check memoization
            if (i, free) in memo:
                return memo[(i, free)]
            
            # If we can take this fruit for free
            if free > 0:
                # Two options: take for free or purchase
                take_free = dp(i+1, free-1)
                purchase = prices[i-1] + dp(i+1, i)
                memo[(i, free)] = min(take_free, purchase)
            else:
                # Must purchase
                memo[(i, free)] = prices[i-1] + dp(i+1, i)
            
            return memo[(i, free)]
        
        # Start from the first fruit with no free fruits
        return dp(1, 0)