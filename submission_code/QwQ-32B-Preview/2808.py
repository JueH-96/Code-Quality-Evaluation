from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # Create a list of walls with their cost and time
        walls = sorted(zip(cost, time), key=lambda x: x[0])
        
        # Compute prefix sums of time
        prefix_time = [0] * (n + 1)
        for i in range(n):
            prefix_time[i + 1] = prefix_time[i] + walls[i][1]
        
        min_cost = float('inf')
        # Iterate over k from 0 to n
        for k in range(n + 1):
            if k == 0:
                if 0 >= n - k:
                    min_cost = 0
                    break
                else:
                    continue
            time_sum = prefix_time[k]
            if time_sum >= n - k:
                cost_sum = sum(walls[i][0] for i in range(k))
                min_cost = min(min_cost, cost_sum)
        
        return min_cost