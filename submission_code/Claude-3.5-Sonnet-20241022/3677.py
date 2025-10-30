class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] represents max coins at position (i,j) with k neutralizations left
        dp = {}
        
        def dfs(i: int, j: int, neutralize: int) -> int:
            # Base cases
            if i >= m or j >= n:
                return float('-inf')
            if i == m-1 and j == n-1:
                # For the final cell, use neutralization if it's negative and we have neutralizations left
                if coins[i][j] < 0 and neutralize > 0:
                    return 0
                return coins[i][j]
            
            # Check if state has been computed
            state = (i, j, neutralize)
            if state in dp:
                return dp[state]
            
            # Current cell value considering neutralization
            current = coins[i][j]
            max_coins = float('-inf')
            
            # Try both paths (right and down) with and without using neutralization
            if current < 0 and neutralize > 0:
                # Try using neutralization
                # Move right
                max_coins = max(max_coins, 
                              dfs(i, j+1, neutralize-1))
                # Move down
                max_coins = max(max_coins, 
                              dfs(i+1, j, neutralize-1))
            
            # Try without using neutralization
            # Move right
            max_coins = max(max_coins, 
                          current + dfs(i, j+1, neutralize))
            # Move down
            max_coins = max(max_coins, 
                          current + dfs(i+1, j, neutralize))
            
            dp[state] = max_coins
            return max_coins
        
        result = dfs(0, 0, 2)
        return result