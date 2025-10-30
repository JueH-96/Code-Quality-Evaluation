from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        m = n // 2
        # Initialize the DP for the first pair (0, n-1)
        prev_dp = {}
        i = 0
        j = n - 1 - i
        for c1 in range(3):
            for c2 in range(3):
                if c1 != c2:
                    prev_dp[(c1, c2)] = cost[i][c1] + cost[j][c2]
        
        # Process the remaining pairs
        for pair_idx in range(1, m):
            current_dp = {}
            i = pair_idx
            j = n - 1 - i
            for (prev_c1, prev_c2), prev_total in prev_dp.items():
                # Iterate all possible current colors for the current pair
                for curr_c1 in range(3):
                    if curr_c1 == prev_c1:
                        continue
                    for curr_c2 in range(3):
                        if curr_c2 == prev_c2 or curr_c1 == curr_c2:
                            continue
                        new_total = prev_total + cost[i][curr_c1] + cost[j][curr_c2]
                        key = (curr_c1, curr_c2)
                        if key not in current_dp or new_total < current_dp[key]:
                            current_dp[key] = new_total
            prev_dp = current_dp
        
        return min(prev_dp.values())