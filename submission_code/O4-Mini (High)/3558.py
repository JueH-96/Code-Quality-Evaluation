from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        # We need to ensure the total "unsafe" cells visited (grid[i][j]==1)
        # does not exceed health-1, so that we end with health>=1.
        max_cost = health - 1
        
        # If the start cell itself is unsafe and exceeds our budget, fail immediately
        start_cost = grid[0][0]
        if start_cost > max_cost:
            return False
        
        # dist[i][j] = minimum number of unsafe cells needed to reach (i,j)
        INF = 10**9
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = start_cost
        
        dq = deque()
        dq.append((0, 0))
        
        while dq:
            i, j = dq.popleft()
            curr_cost = dist[i][j]
            
            # If we've reached the target with an acceptable cost, succeed
            if i == m - 1 and j == n - 1:
                if curr_cost <= max_cost:
                    return True
                # otherwise keep searching for possible better paths
            
            # Explore neighbors
            for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = curr_cost + grid[ni][nj]
                    # Only proceed if this path is cheaper and within our budget
                    if new_cost < dist[ni][nj] and new_cost <= max_cost:
                        dist[ni][nj] = new_cost
                        # 0-1 BFS: push front if it's a safe cell, back if unsafe
                        if grid[ni][nj] == 0:
                            dq.appendleft((ni, nj))
                        else:
                            dq.append((ni, nj))
        
        # If we exhaust the queue without a valid path, return False
        return False