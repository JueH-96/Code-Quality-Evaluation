from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        dp = {frozenset(): 0}
        
        for i in range(n):
            distinct_vals = set(grid[i])
            new_dp = dp.copy()
            for set_val, total_val in dp.items():
                for val in distinct_vals:
                    if val in set_val:
                        continue
                    new_set = set_val | {val}
                    new_total = total_val + val
                    if new_set not in new_dp or new_total > new_dp[new_set]:
                        new_dp[new_set] = new_total
            dp = new_dp
        
        return max(dp.values()) if dp else 0