import math
from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        
        # dp_prev[j][k] stores the min cost for the previous pair of houses,
        # with its left house painted color j and its right house painted color k.
        # Colors are 0-indexed (0, 1, 2).
        dp_prev = [[float('inf')] * 3 for _ in range(3)]

        # Base case: i = 0, for the pair of houses (0, n-1)
        # The cost is simply the sum of painting costs for the chosen colors.
        for j in range(3):
            for k in range(3):
                # Equidistant constraint: color(0) != color(n-1)
                if j != k:
                    dp_prev[j][k] = cost[0][j] + cost[n - 1][k]

        # Iterate through the pairs from i = 1 to n/2 - 1
        for i in range(1, n // 2):
            dp_curr = [[float('inf')] * 3 for _ in range(3)]
            
            # For each potential color pair (j, k) for the current houses (i, n-1-i)...
            for j in range(3):  # color for house i
                for k in range(3): # color for house n-1-i
                    # Equidistant constraint: color(i) != color(n-1-i)
                    if j == k:
                        continue
                    
                    # Find the minimum cost from the previous step (i-1) that allows this transition.
                    # The previous pair was (i-1, n-i).
                    # prev_j is color for house i-1, prev_k is color for house n-i.
                    min_prev_cost = float('inf')
                    for prev_j in range(3):
                        # Adjacency constraint: color(i) != color(i-1)
                        if prev_j == j:
                            continue
                        for prev_k in range(3):
                            # Adjacency constraint: color(n-1-i) != color(n-i)
                            if prev_k == k:
                                continue
                            
                            min_prev_cost = min(min_prev_cost, dp_prev[prev_j][prev_k])
                    
                    # If a valid previous state exists, calculate the cost for the current state.
                    if min_prev_cost != float('inf'):
                        current_pair_cost = cost[i][j] + cost[n - 1 - i][k]
                        dp_curr[j][k] = current_pair_cost + min_prev_cost
            
            # The current DP table becomes the previous one for the next iteration.
            dp_prev = dp_curr

        # The final answer is the minimum value in the last DP table.
        result = float('inf')
        for row in dp_prev:
            result = min(result, min(row))
            
        return result