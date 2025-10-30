from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [{} for _ in range(n + 1)]
        dp[0][0] = 0
        
        for cost_i, time_i in zip(cost, time):
            for k in range(n, -1, -1):
                if not dp[k]:
                    continue
                current_dict = dp[k]
                for t in list(current_dict.keys()):
                    new_k = k + 1
                    if new_k > n:
                        continue
                    new_t = t + time_i
                    new_cost = current_dict[t] + cost_i
                    if new_t not in dp[new_k] or new_cost < dp[new_k].get(new_t, float('inf')):
                        dp[new_k][new_t] = new_cost
        
        min_total = float('inf')
        for k in range(n + 1):
            required = n - k
            current_dict = dp[k]
            if not current_dict:
                continue
            current_min = float('inf')
            for t in current_dict:
                if t >= required:
                    current_min = min(current_min, current_dict[t])
            if current_min < min_total:
                min_total = current_min
        
        return min_total