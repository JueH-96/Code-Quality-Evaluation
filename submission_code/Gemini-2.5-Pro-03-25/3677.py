import math
from typing import List

# Use float('-inf') which is standard in Python for representing negative infinity.

class Solution:
    """
    Solves the robot coin collection problem with robber neutralization using dynamic programming.
    The robot moves from the top-left corner (0, 0) to the bottom-right corner (m-1, n-1)
    of an m x n grid, moving only right or down. Each cell contains coins (>= 0) or a robber (< 0).
    The robot can neutralize at most 2 robbers along its path.
    """
    def maximumAmount(self, coins: List[List[int]]) -> int:
        """
        Calculates the maximum amount of coins the robot can collect.

        Args:
            coins: An m x n grid where coins[i][j] is the coin value (>= 0) or robber amount (< 0).
                   The absolute value of a negative number represents the amount stolen by the robber.

        Returns:
            The maximum profit the robot can gain. Returns -1 if the destination cell (m-1, n-1) 
            is determined to be unreachable under the given constraints (which should typically not happen
            if m >= 1 and n >= 1, but handled for robustness).
        """
        m = len(coins)
        n = len(coins[0])
        
        # dp[r][c][k] stores the maximum coins collected upon reaching cell (r, c) 
        # having used exactly k neutralizations along the path from (0, 0) to (r, c).
        # k can range from 0 to 2.
        # Initialize the DP table with negative infinity to represent states that are 
        # unreachable or have not yet been computed with a valid path.
        dp = [[[float('-inf')] * 3 for _ in range(n)] for _ in range(m)]

        # Base case: Initialize the starting cell (0, 0).
        v = coins[0][0] # Value at the starting cell
        if v >= 0:
            # If the start cell has non-negative coins, collect them. Uses 0 neutralizations.
            dp[0][0][0] = v
        else: # Robber at the starting cell
            # Option 1: Don't neutralize. Robot loses |v| coins. Uses 0 neutralizations.
            dp[0][0][0] = v 
            # Option 2: Neutralize the robber. Robot loses 0 coins. Uses 1 neutralization.
            # This is possible only if we have neutralization charges available (k=1 is within 0..2 range).
            dp[0][0][1] = 0 
            # Using 2 neutralizations at the very first cell isn't possible. dp[0][0][2] remains -inf.

        # Fill the DP table by iterating through each cell of the grid.
        # The order ensures that when computing dp[r][c][k], the values for the 
        # prerequisite cells (dp[r-1][c] and dp[r][c-1]) have already been computed.
        for r in range(m):
            for c in range(n):
                # Skip the base case cell (0, 0) as it's already initialized.
                if r == 0 and c == 0:
                    continue
                
                v = coins[r][c] # Value at the current cell (r, c)

                # Iterate through the possible number of neutralizations used (k = 0, 1, 2) 
                # to reach the current cell (r, c).
                for k in range(3): 
                    
                    # Determine the maximum profit from the paths leading to the current cell (r, c).
                    # A path could arrive from the cell above (r-1, c) or from the cell to the left (r, c-1).

                    # Calculate max profit from previous states assuming the path ended with k neutralizations *before* processing cell (r, c).
                    max_prev_k = float('-inf')
                    if r > 0: 
                        max_prev_k = max(max_prev_k, dp[r-1][c][k]) # Max profit from top with k neutralizations
                    if c > 0: 
                        max_prev_k = max(max_prev_k, dp[r][c-1][k]) # Max profit from left with k neutralizations

                    # Calculate max profit from previous states assuming the path ended with k-1 neutralizations *before* processing cell (r, c).
                    # This is relevant if we neutralize a robber at the current cell (r, c).
                    max_prev_k_minus_1 = float('-inf')
                    if k > 0: # Neutralization is only possible if we have used k-1 charges before and k > 0.
                        if r > 0: 
                            max_prev_k_minus_1 = max(max_prev_k_minus_1, dp[r-1][c][k-1]) # Max profit from top with k-1 neutralizations
                        if c > 0: 
                            max_prev_k_minus_1 = max(max_prev_k_minus_1, dp[r][c-1][k-1]) # Max profit from left with k-1 neutralizations

                    # Calculate the maximum profit for state dp[r][c][k] based on the current cell's value 'v'.
                    current_val = float('-inf') # Initialize the potential value for dp[r][c][k]

                    if v >= 0: # Cell has non-negative coins (or zero)
                        # Collect the coins. The number of neutralizations used (k) remains the same.
                        # The path must have arrived from a state using k neutralizations.
                        if max_prev_k != float('-inf'): # Check if the previous state k was reachable
                             current_val = max_prev_k + v
                    else: # Cell has a robber (v < 0)
                        # Option 1: Don't neutralize the robber at (r, c).
                        # Total neutralizations used remains k. Path must come from a state with k neutralizations.
                        # Profit = (max profit from previous cell with k neutralizations) + v (loss)
                        profit_without_neutralizing = float('-inf')
                        if max_prev_k != float('-inf'):
                            profit_without_neutralizing = max_prev_k + v

                        # Option 2: Neutralize the robber at (r, c).
                        # This uses one neutralization charge. Total neutralizations used becomes k.
                        # Path must come from a state with k-1 neutralizations.
                        # Profit = (max profit from previous cell with k-1 neutralizations) + 0 (no loss due to neutralization)
                        profit_with_neutralizing = float('-inf')
                        # We can only neutralize if k > 0 (i.e., we are calculating for state k=1 or k=2) 
                        # and if the previous state (k-1) was reachable.
                        if k > 0 and max_prev_k_minus_1 != float('-inf'):
                            profit_with_neutralizing = max_prev_k_minus_1 # Add 0 for the neutralized robber's effect

                        # The max profit for dp[r][c][k] is the maximum of the two options (neutralize or not).
                        current_val = max(profit_without_neutralizing, profit_with_neutralizing)

                    # Update the DP table entry for (r, c) with k neutralizations.
                    # We assign current_val which already incorporates the maximum profit from valid incoming paths.
                    dp[r][c][k] = current_val

        # The final answer is the maximum profit achievable at the destination cell (m-1, n-1)
        # across all possible states of neutralization count (k=0, 1, or 2).
        final_result = max(dp[m - 1][n - 1]) 

        # According to the problem constraints (m, n >= 1), a path always exists.
        # The calculated maximum profit should always be a finite number (possibly negative).
        # If final_result remains -infinity, it indicates an issue, potentially suggesting
        # that the destination is unreachable under the 2-neutralization limit (e.g., if all paths require > 2).
        # Return -1 in this scenario, following common practice for unreachable targets in pathfinding problems.
        if final_result == float('-inf'):
             return -1 
        
        # Otherwise, return the computed maximum profit, cast to an integer.
        return int(final_result)