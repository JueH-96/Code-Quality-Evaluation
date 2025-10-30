from typing import List
from collections import deque
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Precompute the distance from a start point to all other points using BFS
        def bfs_all(start):
            visited = [[-1 for _ in range(50)] for _ in range(50)]
            queue = deque([(start[0], start[1])])
            visited[start[0]][start[1]] = 0
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        queue.append((nx, ny))
            return visited

        # Prepare the list of points, including knight's initial position and all pawns
        n = len(positions)
        points = [(kx, ky)] + positions

        # Precompute the distance matrix
        distance = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dist_grid = bfs_all(points[i])
            for j in range(n + 1):
                x, y = points[j]
                distance[i][j] = dist_grid[x][y]

        initial_mask = (1 << n) - 1  # All pawns are present initially

        @lru_cache(maxsize=None)
        def dfs(current_point, mask):
            if mask == 0:
                return 0
            captured_count = n - bin(mask).count('1')
            is_alice_turn = (captured_count % 2) == 0

            max_sum = -float('inf')
            min_sum = float('inf')

            for j in range(1, n + 1):
                bit = j - 1
                if not (mask & (1 << bit)):
                    continue
                dist = distance[current_point][j]
                new_mask = mask & ~ (1 << bit)
                current_sum = dist + dfs(j, new_mask)
                if is_alice_turn:
                    if current_sum > max_sum:
                        max_sum = current_sum
                else:
                    if current_sum < min_sum:
                        min_sum = current_sum

            return max_sum if is_alice_turn else min_sum

        return dfs(0, initial_mask)