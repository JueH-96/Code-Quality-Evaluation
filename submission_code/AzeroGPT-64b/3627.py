from typing import List
from heapq import heappop, heappush

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        
        rows, cols = len(moveTime), len(moveTime[0])
        visited = set()
        min_heap = [(moveTime[0][0], 0, 0)]
        
        while min_heap:
            time, row, col = heappop(min_heap)
            if (row, col) in visited: continue
            visited.add((row, col))
            if row == rows-1 and col == cols-1: return time
            for r, c in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited:
                    heappush(min_heap, (max(time, moveTime[r][c]) + 1, r, c))
        
        return -1