import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        if m == 0:
            return False
        n = len(grid[0])
        if m * n == 1:
            return True
        
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        cost = [[float('inf') for _ in range(n)] for _ in range(m)]
        cost[0][0] = 0
        
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        
        while heap:
            current_cost, i, j = heapq.heappop(heap)
            
            if i == m - 1 and j == n - 1:
                if current_cost <= health:
                    return True
                else:
                    continue
            
            if current_cost > cost[i][j]:
                continue
            
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = current_cost + grid[ni][nj]
                    if new_cost < cost[ni][nj]:
                        cost[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        
        return False