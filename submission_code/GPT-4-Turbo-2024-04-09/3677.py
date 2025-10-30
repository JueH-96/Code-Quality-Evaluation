class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        from functools import lru_cache
        
        m, n = len(coins), len(coins[0])
        
        @lru_cache(None)
        def dp(x, y, neutralizations):
            if x >= m or y >= n:
                return float('-inf')  # out of bounds
            if x == m - 1 and y == n - 1:
                # Last cell, apply neutralization if possible and needed
                if coins[x][y] < 0 and neutralizations > 0:
                    return -coins[x][y]
                else:
                    return coins[x][y]
            
            current_coins = coins[x][y]
            if current_coins < 0 and neutralizations > 0:
                # Option to neutralize this cell
                neutralize = -current_coins + max(dp(x + 1, y, neutralizations - 1), dp(x, y + 1, neutralizations - 1))
            else:
                neutralize = float('-inf')
            
            # Option to not neutralize this cell
            not_neutralize = current_coins + max(dp(x + 1, y, neutralizations), dp(x, y + 1, neutralizations))
            
            return max(neutralize, not_neutralize)
        
        return dp(0, 0, 2)