from typing import List
from collections import deque

class Solution:
    _dist = None

    @classmethod
    def precompute_dist(cls):
        if cls._dist is not None:
            return
        # Initialize 4D list for distances
        dist = [[[[0 for _ in range(50)] for _ in range(50)] for _ in range(50)] for _ in range(50)]
        dx = [2, 2, -2, -2, 1, 1, -1, -1]
        dy = [1, -1, 1, -1, 2, -2, 2, -2]
        for x in range(50):
            for y in range(50):
                # Reset all distances to -1 for this (x, y)
                for a in range(50):
                    for b in range(50):
                        dist[x][y][a][b] = -1
                # BFS from (x, y)
                q = deque()
                q.append((x, y))
                dist[x][y][x][y] = 0
                while q:
                    cx, cy = q.popleft()
                    for i in range(8):
                        nx = cx + dx[i]
                        ny = cy + dy[i]
                        if 0 <= nx < 50 and 0 <= ny < 50:
                            if dist[x][y][nx][ny] == -1:
                                dist[x][y][nx][ny] = dist[x][y][cx][cy] + 1
                                q.append((nx, ny))
        cls._dist = dist

    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        self.precompute_dist()
        nodes = [(kx, ky)] + positions
        n = len(positions)
        # Compute distance matrix between nodes
        dist_nodes = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            x1, y1 = nodes[i]
            for j in range(n + 1):
                x2, y2 = nodes[j]
                dist_nodes[i][j] = self._dist[x1][y1][x2][y2]
        from functools import lru_cache
        @lru_cache(None)
        def dfs(current_node, mask, is_alice_turn):
            if mask == 0:
                return 0
            res = None
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                steps = dist_nodes[current_node][i + 1]
                new_mask = mask & ~(1 << i)
                total = steps + dfs(i + 1, new_mask, not is_alice_turn)
                if res is None:
                    res = total
                else:
                    if is_alice_turn:
                        res = max(res, total)
                    else:
                        res = min(res, total)
            return res
        initial_mask = (1 << n) - 1
        return dfs(0, initial_mask, True)