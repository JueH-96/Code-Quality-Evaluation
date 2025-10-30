from typing import List
import heapq

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        min_cost = [float('inf')] * (n + 1)
        min_cost[0] = 0
        
        for i in range(n):
            # Use the paid painter for the i-th wall
            for j in range(i + 1):
                if j + time[i] <= n:
                    min_cost[j + time[i]] = min(min_cost[j + time[i]], min_cost[j] + cost[i])
        
        return min(min_cost)

# Example usage:
# sol = Solution()
# print(sol.paintWalls([1,2,3,2], [1,2,3,2]))  # Output: 3
# print(sol.paintWalls([2,3,4,2], [1,1,1,1]))  # Output: 4