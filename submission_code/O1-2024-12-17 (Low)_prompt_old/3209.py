class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        from functools import lru_cache
        
        n = len(prices)
        
        @lru_cache(None)
        def dfs(i, free):
            # If we've obtained all fruits, cost is 0
            if i > n:
                return 0
            
            # If current fruit i is free (free > 0), we can skip its cost
            # but we can still consider buying it if that leads to a better overall outcome
            if free > 0:
                skip_cost = dfs(i + 1, free - 1)
                buy_cost = prices[i - 1] + dfs(i + 1, i)  # buying i gives i free fruits
                return min(skip_cost, buy_cost)
            else:
                # If no free fruits left, we must buy fruit i
                return prices[i - 1] + dfs(i + 1, i)
        
        # Start from the first fruit with 0 free fruits
        return dfs(1, 0)