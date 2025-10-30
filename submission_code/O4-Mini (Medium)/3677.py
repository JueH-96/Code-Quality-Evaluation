from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        # Use a large negative number as -infinity
        NEG_INF = -10**18
        
        # dp for previous row, each entry is a list of size 3 for k=0,1,2 neutralizations used
        prev_dp = [[NEG_INF] * 3 for _ in range(n)]
        
        # Initialize row 0, column 0
        c0 = coins[0][0]
        if c0 >= 0:
            prev_dp[0][0] = c0
            # cannot sensibly use a neutralization on a positive cell
        else:
            # either take the hit or neutralize here
            prev_dp[0][0] = c0
            prev_dp[0][1] = 0
        
        # Fill the rest of row 0 (only from left)
        for j in range(1, n):
            c = coins[0][j]
            for k in range(3):
                if prev_dp[j-1][k] == NEG_INF:
                    # unreachable
                    continue
                if c >= 0:
                    # just add coins
                    prev_dp[j][k] = prev_dp[j-1][k] + c
                else:
                    # either take the robbery
                    val_without = prev_dp[j-1][k] + c
                    best = val_without
                    # or neutralize if we have budget
                    if k + 1 < 3:
                        val_neut = prev_dp[j-1][k]
                        best = max(best, val_neut)
                        # that would use one more neutralization
                        # but we store exact k+1 in that slot below
                        prev_dp[j][k+1] = max(prev_dp[j][k+1], val_neut)
                    prev_dp[j][k] = max(prev_dp[j][k], best)
        
        # Process rows 1..m-1
        for i in range(1, m):
            # curr_dp for this row
            curr_dp = [[NEG_INF] * 3 for _ in range(n)]
            
            # first column of this row (only from top)
            c = coins[i][0]
            for k in range(3):
                if prev_dp[0][k] == NEG_INF:
                    continue
                if c >= 0:
                    curr_dp[0][k] = prev_dp[0][k] + c
                else:
                    val_without = prev_dp[0][k] + c
                    best = val_without
                    if k + 1 < 3:
                        val_neut = prev_dp[0][k]
                        best = max(best, val_neut)
                        curr_dp[0][k+1] = max(curr_dp[0][k+1], val_neut)
                    curr_dp[0][k] = max(curr_dp[0][k], best)
            
            # fill rest of columns j=1..n-1
            for j in range(1, n):
                c = coins[i][j]
                for k in range(3):
                    # consider coming from top or left without using new neutralization
                    best_prev = NEG_INF
                    if prev_dp[j][k] != NEG_INF:
                        best_prev = max(best_prev, prev_dp[j][k])
                    if curr_dp[j-1][k] != NEG_INF:
                        best_prev = max(best_prev, curr_dp[j-1][k])
                    if best_prev != NEG_INF:
                        if c >= 0:
                            curr_dp[j][k] = max(curr_dp[j][k], best_prev + c)
                        else:
                            curr_dp[j][k] = max(curr_dp[j][k], best_prev + c)
                    # if it's a robber cell, consider using one more neutralization
                    if c < 0 and k > 0:
                        best_prev_neut = NEG_INF
                        if prev_dp[j][k-1] != NEG_INF:
                            best_prev_neut = max(best_prev_neut, prev_dp[j][k-1])
                        if curr_dp[j-1][k-1] != NEG_INF:
                            best_prev_neut = max(best_prev_neut, curr_dp[j-1][k-1])
                        if best_prev_neut != NEG_INF:
                            # neutralize here => no gain/loss in this cell
                            curr_dp[j][k] = max(curr_dp[j][k], best_prev_neut)
            
            # move to next row
            prev_dp = curr_dp
        
        # answer is max over using 0,1,2 neutralizations at bottom-right
        return max(prev_dp[n-1])