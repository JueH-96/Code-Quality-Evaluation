from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        INF = float('-inf')
        
        # Initialize previous row (for row 0)
        prev_row = [[INF] * 3 for _ in range(n)]
        
        # Base case for (0,0)
        c00 = coins[0][0]
        prev_row[0][0] = c00
        prev_row[0][1] = max(0, c00)
        prev_row[0][2] = max(0, c00)
        
        # Fill the first row (i=0)
        for j in range(1, n):
            for k in range(3):
                current_max = INF
                # Can only come from left (j-1)
                # Option 1: not use neutralization here
                if prev_row[j-1][k] != INF:
                    candidate = prev_row[j-1][k] + coins[0][j]
                    if candidate > current_max:
                        current_max = candidate
                # Option 2: use neutralization here (if possible)
                if k >= 1 and prev_row[j-1][k-1] != INF:
                    candidate = prev_row[j-1][k-1] + max(0, coins[0][j])
                    if candidate > current_max:
                        current_max = candidate
                prev_row[j][k] = current_max
        
        # Process remaining rows
        for i in range(1, m):
            current_row = [[INF] * 3 for _ in range(n)]
            for j in range(n):
                if j == 0:
                    # Can only come from top (i-1,0)
                    for k in range(3):
                        current_max = INF
                        # from top (prev_row[0][k])
                        # option 1: not use neutralization here
                        if prev_row[0][k] != INF:
                            candidate = prev_row[0][k] + coins[i][0]
                            if candidate > current_max:
                                current_max = candidate
                        # option 2: use neutralization here (if possible)
                        if k >= 1 and prev_row[0][k-1] != INF:
                            candidate = prev_row[0][k-1] + max(0, coins[i][0])
                            if candidate > current_max:
                                current_max = candidate
                        current_row[0][k] = current_max
                else:
                    # can come from top or left
                    for k in range(3):
                        current_max = INF
                        # from top (prev_row[j][k])
                        if prev_row[j][k] != INF:
                            candidate = prev_row[j][k] + coins[i][j]
                            if candidate > current_max:
                                current_max = candidate
                        if k >= 1 and prev_row[j][k-1] != INF:
                            candidate = prev_row[j][k-1] + max(0, coins[i][j])
                            if candidate > current_max:
                                current_max = candidate
                        # from left (current_row[j-1][k])
                        if current_row[j-1][k] != INF:
                            candidate = current_row[j-1][k] + coins[i][j]
                            if candidate > current_max:
                                current_max = candidate
                        if k >= 1 and current_row[j-1][k-1] != INF:
                            candidate = current_row[j-1][k-1] + max(0, coins[i][j])
                            if candidate > current_max:
                                current_max = candidate
                        current_row[j][k] = current_max
            prev_row = current_row
        
        # The final answer is the maximum of the three possibilities at the last cell
        return max(prev_row[-1][0], prev_row[-1][1], prev_row[-1][2])