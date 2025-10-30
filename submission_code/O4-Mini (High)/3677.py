from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        # maximum neutralizations allowed
        K = 2
        # a very small number to represent -infinity
        NEG_INF = -10**18
        
        # dp_prev[j][t] = max coins collected reaching cell (i-1, j)
        #                having used exactly t neutralizations
        dp_prev = None
        
        for i in range(m):
            # dp_cur[j][t] for the current row i
            dp_cur = [[NEG_INF] * (K+1) for _ in range(n)]
            for j in range(n):
                val = coins[i][j]
                
                # starting cell
                if i == 0 and j == 0:
                    # if it's non-negative, just take it, no neutralization used
                    if val >= 0:
                        dp_cur[0][0] = val
                    else:
                        # take it without neutralizing
                        dp_cur[0][0] = val
                        # or neutralize it (using 1 of our K=2 neutralizations)
                        dp_cur[0][1] = 0
                    continue
                
                # fill dp_cur[j][t] for t = 0..2
                for t in range(K+1):
                    if val >= 0:
                        # no need (and can't) neutralize a non-negative cell
                        best_prev = NEG_INF
                        # from top (i-1, j)
                        if i > 0:
                            prev = dp_prev[j][t]
                            if prev > best_prev:
                                best_prev = prev
                        # from left (i, j-1)
                        if j > 0:
                            prev = dp_cur[j-1][t]
                            if prev > best_prev:
                                best_prev = prev
                        if best_prev > NEG_INF:
                            dp_cur[j][t] = best_prev + val
                    else:
                        # two choices: don't neutralize (and lose |val|),
                        # or neutralize here (if t>0) and lose nothing.
                        best_val = NEG_INF
                        # 1) don't neutralize: use exactly t so far
                        best_not = NEG_INF
                        if i > 0:
                            prev = dp_prev[j][t]
                            if prev > best_not:
                                best_not = prev
                        if j > 0:
                            prev = dp_cur[j-1][t]
                            if prev > best_not:
                                best_not = prev
                        if best_not > NEG_INF:
                            best_val = best_not + val
                        # 2) neutralize here: uses one neutralization
                        if t > 0:
                            best_neu = NEG_INF
                            if i > 0:
                                prev = dp_prev[j][t-1]
                                if prev > best_neu:
                                    best_neu = prev
                            if j > 0:
                                prev = dp_cur[j-1][t-1]
                                if prev > best_neu:
                                    best_neu = prev
                            if best_neu > NEG_INF and best_neu > best_val:
                                best_val = best_neu
                        dp_cur[j][t] = best_val
            
            # move current row to previous for next iteration
            dp_prev = dp_cur
        
        # answer is the best we can do at bottom-right (m-1,n-1) using up to 2 neutralizations
        return max(dp_prev[n-1])