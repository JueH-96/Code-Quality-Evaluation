import math
from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # dp_current[c1][c2] will store the minimum cost to paint houses
        # from k to n-1-k, where house k has color c1 and house n-1-k has color c2.
        # The loop variable k represents the current house index from the left,
        # ranging from 0 to n/2 - 1.
        # c1 and c2 are 0-indexed colors (0, 1, 2)
        
        # Initialize dp_current for k = 0 (houses 0 and n-1)
        # dp_current will store costs for the current pair (k, n-1-k)
        dp_current = [[math.inf] * 3 for _ in range(3)]
        
        # Base case: k = 0. This involves houses 0 and n-1.
        # cost[0] is for house 0, cost[n-1] is for house n-1.
        for c1 in range(3):
            for c2 in range(3):
                # Condition: houses at positions 0 and n-1 (equidistant from ends)
                # must not be painted the same color.
                if c1 != c2:
                    dp_current[c1][c2] = cost[0][c1] + cost[n-1][c2]
        
        # Iterate for k from 1 to n/2 - 1.
        # Each iteration processes a new pair of equidistant houses (k, n-1-k).
        for k in range(1, n // 2):
            # dp_prev stores the results from the previous iteration (k-1).
            # These are costs for houses (k-1, n-k).
            dp_prev = dp_current
            # dp_current will store results for the current iteration (k).
            # These are costs for houses (k, n-1-k).
            dp_current = [[math.inf] * 3 for _ in range(3)]
            
            # Current houses being painted are 'k' (from left) and 'n-1-k' (from right).
            # Let their colors be c1 and c2 respectively.
            
            # The previously painted houses relevant for adjacency are 'k-1' and 'n-1-(k-1)' which is 'n-k'.
            # Let their colors be prev_c1 and prev_c2 respectively.
            
            for c1 in range(3): # Color for house k
                for c2 in range(3): # Color for house n-1-k
                    # Condition 1: Houses equidistant from ends must not be painted the same color.
                    # This applies to the current pair (k, n-1-k).
                    if c1 == c2:
                        continue 
                    
                    # Calculate the cost of painting the current pair of houses with c1 and c2.
                    current_pair_cost = cost[k][c1] + cost[n-1-k][c2]
                    
                    min_prev_state_cost = math.inf
                    
                    # Iterate over all possible color combinations for the previous pair (k-1, n-k).
                    for prev_c1 in range(3): # Color for house k-1
                        for prev_c2 in range(3): # Color for house n-k
                            # Condition 2: Adjacent houses must not be painted the same color.
                            # Applies to house k-1 and house k.
                            if prev_c1 == c1:
                                continue
                            
                            # Condition 3: Adjacent houses must not be painted the same color.
                            # Applies to house n-k and house n-1-k.
                            if prev_c2 == c2:
                                continue
                            
                            # Check if the previous state (prev_c1, prev_c2) was reachable (not infinity).
                            # If it was reachable, consider adding its cost to form the current state.
                            if dp_prev[prev_c1][prev_c2] != math.inf:
                                min_prev_state_cost = min(min_prev_state_cost, dp_prev[prev_c1][prev_c2])
                    
                    # If a valid path to the current state (c1, c2) was found from any previous state,
                    # update the current DP value.
                    if min_prev_state_cost != math.inf:
                        dp_current[c1][c2] = current_pair_cost + min_prev_state_cost
        
        # After iterating through all pairs, the minimum total cost will be the minimum value
        # in the final dp_current table (which represents costs up to pair (n/2 - 1, n/2)).
        min_total_cost = math.inf
        for c1 in range(3):
            for c2 in range(3):
                min_total_cost = min(min_total_cost, dp_current[c1][c2])
                
        return min_total_cost