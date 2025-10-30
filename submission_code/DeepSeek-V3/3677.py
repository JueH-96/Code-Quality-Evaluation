class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0]) if m > 0 else 0
        # DP[i][j][k] represents max coins at (i,j) with k neutralizations used (k can be 0,1,2)
        dp = [[[-1 for _ in range(3)] for __ in range(n)] for ___ in range(m)]
        
        # Initialize the starting cell
        for k in range(3):
            val = coins[0][0]
            if val < 0:
                if k >= 1:
                    dp[0][0][k] = 0  # neutralize it, so gain 0
                else:
                    dp[0][0][k] = val  # lose the coins
            else:
                dp[0][0][k] = val
        
        # Fill the first row: only can move right
        for j in range(1, n):
            for k in range(3):
                max_val = -1
                # Only from left
                if dp[0][j-1][k] != -1:
                    current = coins[0][j]
                    if current < 0:
                        # Option 1: don't neutralize if possible (k remains same)
                        option1 = dp[0][j-1][k] + current
                        # Option 2: neutralize if k > 0
                        option2 = -1
                        if k >= 1:
                            option2 = dp[0][j-1][k-1] + 0
                        max_val = max(option1, option2)
                    else:
                        max_val = dp[0][j-1][k] + current
                dp[0][j][k] = max_val
        
        # Fill the first column: only can move down
        for i in range(1, m):
            for k in range(3):
                max_val = -1
                # Only from top
                if dp[i-1][0][k] != -1:
                    current = coins[i][0]
                    if current < 0:
                        option1 = dp[i-1][0][k] + current
                        option2 = -1
                        if k >= 1:
                            option2 = dp[i-1][0][k-1] + 0
                        max_val = max(option1, option2)
                    else:
                        max_val = dp[i-1][0][k] + current
                dp[i][0][k] = max_val
        
        # Fill the rest of the table
        for i in range(1, m):
            for j in range(1, n):
                for k in range(3):
                    max_val = -1
                    # Check top and left
                    candidates = []
                    if dp[i-1][j][k] != -1:
                        candidates.append(dp[i-1][j][k])
                    if dp[i][j-1][k] != -1:
                        candidates.append(dp[i][j-1][k])
                    if not candidates:
                        continue
                    best_prev = max(candidates)
                    current = coins[i][j]
                    if current < 0:
                        # Option 1: don't neutralize
                        option1 = best_prev + current
                        # Option 2: neutralize if possible
                        option2 = -1
                        if k >= 1:
                            # Find the best previous with k-1 neutralizations
                            candidates_k_minus_1 = []
                            if dp[i-1][j][k-1] != -1:
                                candidates_k_minus_1.append(dp[i-1][j][k-1])
                            if dp[i][j-1][k-1] != -1:
                                candidates_k_minus_1.append(dp[i][j-1][k-1])
                            if candidates_k_minus_1:
                                best_prev_k_minus_1 = max(candidates_k_minus_1)
                                option2 = best_prev_k_minus_1 + 0
                        max_val = max(option1, option2)
                    else:
                        max_val = best_prev + current
                    dp[i][j][k] = max_val
        
        result = max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])
        return max(result, -1)  # in case all paths are invalid, but constraints say m,n >=1