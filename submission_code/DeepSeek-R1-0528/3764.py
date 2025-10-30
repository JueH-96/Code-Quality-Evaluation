import heapq
from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        m = len(grid[0])
        arrs = []
        for i in range(n):
            row = grid[i][:]
            row.sort(reverse=True)
            take = min(limits[i], m)
            arrs.append(row[:take])
        
        heap = []
        for i in range(n):
            if arrs[i]:
                heapq.heappush(heap, (-arrs[i][0], i, 0))
        
        total = 0
        for _ in range(k):
            neg_val, i, idx = heapq.heappop(heap)
            total += -neg_val
            if idx + 1 < len(arrs[i]):
                heapq.heappush(heap, (-arrs[i][idx+1], i, idx+1))
        
        return total