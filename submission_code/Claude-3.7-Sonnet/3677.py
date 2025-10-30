class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # Using memoization
        memo = {}
        
        def dp(i, j, k):
            # Out of bounds
            if i >= m or j >= n:
                return float('-inf')
            
            # If we've already computed this state
            if (i, j, k) in memo:
                return memo[(i, j, k)]
            
            # Base case: at the destination
            if i == m - 1 and j == n - 1:
                if coins[i][j] < 0 and k > 0:
                    # Either use neutralization or not
                    return max(0, coins[i][j])
                else:
                    return coins[i][j]
            
            # Recursive case
            result = float('-inf')
            
            # Option 1: Don't use neutralization for the current cell
            # Move right without neutralization
            if j + 1 < n:
                result = max(result, coins[i][j] + dp(i, j + 1, k))
            
            # Move down without neutralization
            if i + 1 < m:
                result = max(result, coins[i][j] + dp(i + 1, j, k))
            
            # Option 2: Use neutralization for the current cell (if applicable)
            if coins[i][j] < 0 and k > 0:
                # Move right with neutralization
                if j + 1 < n:
                    result = max(result, 0 + dp(i, j + 1, k - 1))
                
                # Move down with neutralization
                if i + 1 < m:
                    result = max(result, 0 + dp(i + 1, j, k - 1))
            
            memo[(i, j, k)] = result
            return result
        
        # Start from the top-left with 2 neutralizations
        return dp(0, 0, 2)