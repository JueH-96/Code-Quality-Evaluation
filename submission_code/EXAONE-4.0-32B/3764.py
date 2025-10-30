import heapq
from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        sorted_grid = [sorted(row, reverse=True) for row in grid]
        
        if k == 0:
            return 0
            
        count = [0] * n
        heap = []
        
        for i in range(n):
            if limits[i] > 0:
                heapq.heappush(heap, (-sorted_grid[i][0], i, 0))
                
        total_sum = 0
        selected = 0
        
        while selected < k and heap:
            neg_val, i, idx = heapq.heappop(heap)
            total_sum += -neg_val
            selected += 1
            count[i] += 1
            if count[i] < limits[i] and idx + 1 < len(sorted_grid[i]):
                next_val = sorted_grid[i][idx+1]
                heapq.heappush(heap, (-next_val, i, idx+1))
                
        return total_sum