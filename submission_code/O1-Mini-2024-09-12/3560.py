from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        all_positions = [tuple(pos) for pos in positions]
        start_index = n
        all_positions.append((kx, ky))
        
        # Precompute distances between all pawns and the starting position
        distances = [[-1 for _ in range(n)] for _ in range(n+1)]
        
        directions = [(-2, -1), (-1, -2), (1, -2), (2, -1),
                      (2, 1), (1, 2), (-1, 2), (-2, 1)]
        
        def bfs(start_pos):
            queue = deque()
            queue.append((start_pos[0], start_pos[1], 0))
            visited = [[-1 for _ in range(50)] for _ in range(50)]
            visited[start_pos[0]][start_pos[1]] = 0
            while queue:
                x, y, d = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = d + 1
                        queue.append((nx, ny, d + 1))
            return visited
        
        # Compute all distances
        visited_all = []
        for i in range(n+1):
            visited = bfs(all_positions[i])
            visited_all.append(visited)
        
        for i in range(n+1):
            for j in range(n):
                distances[i][j] = visited_all[i][all_positions[j][0]][all_positions[j][1]]
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(current, mask, is_alice):
            if mask == 0:
                return 0
            if is_alice:
                best = -float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        cost = distances[current][i]
                        if cost == -1:
                            continue
                        total = cost + dfs(i, mask ^ (1 << i), False)
                        if total > best:
                            best = total
                return best
            else:
                best = float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        cost = distances[current][i]
                        if cost == -1:
                            continue
                        total = cost + dfs(i, mask ^ (1 << i), True)
                        if total < best:
                            best = total
                return best
        
        full_mask = (1 << n) - 1
        return dfs(start_index, full_mask, True)