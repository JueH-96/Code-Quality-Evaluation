import math

class Solution:
  def maximumAmount(self, coins: list[list[int]]) -> int:
    m = len(coins)
    n = len(coins[0])

    # dp[r][c][k] = max coins to reach (r,c) using k neutralizations
    # k = 0, 1, or 2
    # Initialize with negative infinity.
    dp = [[[-math.inf] * 3 for _ in range(n)] for _ in range(m)]

    # Base case: cell (0,0)
    val_at_00 = coins[0][0]
    if val_at_00 >= 0:
        # Gained coins, 0 neutralizations used
        dp[0][0][0] = val_at_00
    else: # val_at_00 < 0 (robber)
        # Option 1: Don't neutralize. Lose coins. 0 neutralizations used.
        dp[0][0][0] = val_at_00
        # Option 2: Neutralize. Lose 0 coins from this cell. 1 neutralization used.
        # (Cannot use 2 neutralizations on a single cell, so dp[0][0][2] remains -math.inf)
        if 1 < 3: # Ensure k=1 is a valid index for neutralizations
             dp[0][0][1] = 0
    
    # Iterate through the grid
    for r in range(m):
        for c in range(n):
            current_cell_val = coins[r][c]

            for k_total_used in range(3): # k_total_used = 0, 1, or 2
                
                # Option 1: Current cell (r,c) is NOT neutralized.
                # Score if coming from (r-1,c) or (r,c-1) having already used k_total_used neutralizations.
                # Robot collects current_cell_val.
                res_current_not_neutralized = -math.inf
                
                if r > 0: # Coming from (r-1, c)
                    if dp[r-1][c][k_total_used] != -math.inf:
                        res_current_not_neutralized = max(res_current_not_neutralized, dp[r-1][c][k_total_used] + current_cell_val)
                
                if c > 0: # Coming from (r, c-1)
                    if dp[r][c-1][k_total_used] != -math.inf:
                        res_current_not_neutralized = max(res_current_not_neutralized, dp[r][c-1][k_total_used] + current_cell_val)
                
                # Option 2: Current cell (r,c) IS neutralized.
                # Score if coming from (r-1,c) or (r,c-1) having used k_total_used-1 neutralizations,
                # and using the k_total_used-th neutralization at (r,c).
                # Robot collects 0 from current_cell_val.
                res_current_neutralized = -math.inf
                
                # This option is only valid if current_cell_val is a robber and we have a neutralization to use.
                if current_cell_val < 0 and k_total_used > 0:
                    if r > 0: # Coming from (r-1, c)
                        if dp[r-1][c][k_total_used-1] != -math.inf:
                             res_current_neutralized = max(res_current_neutralized, dp[r-1][c][k_total_used-1]) # +0 for current cell
                    
                    if c > 0: # Coming from (r, c-1)
                        if dp[r][c-1][k_total_used-1] != -math.inf:
                            res_current_neutralized = max(res_current_neutralized, dp[r][c-1][k_total_used-1]) # +0 for current cell
                
                # For (0,0), dp[0][0][k_total_used] is pre-initialized.
                # The paths from r>0 or c>0 won't contribute if r=0,c=0.
                # So, we take max with existing dp[r][c][k_total_used] to preserve base case values.
                # For other cells, dp[r][c][k_total_used] is initially -math.inf.
                current_dp_val = dp[r][c][k_total_used]
                dp[r][c][k_total_used] = max(current_dp_val, res_current_not_neutralized, res_current_neutralized)

    # The result is the maximum coins at cell (m-1, n-1) using any number of allowed neutralizations
    final_ans = -math.inf
    if m > 0 and n > 0 : # Grid is not empty
        final_ans = max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])
    
    # The problem implies a path always exists and a value can be computed.
    # The result can be negative.
    return int(final_ans)