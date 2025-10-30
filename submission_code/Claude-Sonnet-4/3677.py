class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] = maximum coins at position (i,j) with k neutralizations used
        # k can be 0, 1, or 2
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Base case: starting position
        if coins[0][0] < 0:
            # We can either take the robbery or neutralize it
            dp[0][0][0] = coins[0][0]  # Take the robbery
            dp[0][0][1] = 0  # Neutralize the robber
        else:
            dp[0][0][0] = coins[0][0]  # Gain coins
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                for k in range(3):  # k neutralizations used
                    # Coming from top
                    if i > 0 and dp[i-1][j][k] != -float('inf'):
                        if coins[i][j] < 0:
                            # This is a robber cell
                            # Option 1: Take the robbery
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                            # Option 2: Neutralize if we have neutralizations left
                            if k > 0:
                                dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1])
                        else:
                            # This is a coin cell
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                    
                    # Coming from left
                    if j > 0 and dp[i][j-1][k] != -float('inf'):
                        if coins[i][j] < 0:
                            # This is a robber cell
                            # Option 1: Take the robbery
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                            # Option 2: Neutralize if we have neutralizations left
                            if k > 0:
                                dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1])
                        else:
                            # This is a coin cell
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
        
        # Return the maximum among all possible neutralization states at the destination
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])