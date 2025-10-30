from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        # dist[i][j] = minimal health cost to reach (i,j)
        INF = m * n + 5
        dist = [[INF] * n for _ in range(m)]
        
        # 0-1 BFS initialization
        dq = deque()
        dist[0][0] = grid[0][0]
        dq.append((0, 0))
        
        # We only care about paths whose cost < health,
        # since cost must be <= health-1 to have final health >= 1.
        while dq:
            i, j = dq.popleft()
            current_cost = dist[i][j]
            # Early exit if we've already exceeded allowable cost
            if current_cost >= health:
                continue
            for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    w = grid[ni][nj]
                    new_cost = current_cost + w
                    # Only proceed if we find a strictly better cost
                    # and it stays below health
                    if new_cost < dist[ni][nj] and new_cost < health:
                        dist[ni][nj] = new_cost
                        # 0-weight edges go to front, 1-weight to back
                        if w == 0:
                            dq.appendleft((ni, nj))
                        else:
                            dq.append((ni, nj))
        
        # Check if we can reach bottom-right with cost <= health-1
        return dist[m-1][n-1] <= health - 1