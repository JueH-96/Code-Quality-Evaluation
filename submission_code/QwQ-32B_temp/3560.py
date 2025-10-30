import sys
from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        points = [(kx, ky)] + [(x, y) for x, y in positions]
        n_p = len(points)
        
        # Precompute the distance matrix D[i][j] where D[i][j] is the minimal steps from points[i] to points[j]
        D = [[0] * n_p for _ in range(n_p)]
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for i in range(n_p):
            start_x, start_y = points[i]
            dist = [[-1 for _ in range(50)] for _ in range(50)]
            q = deque()
            dist[start_x][start_y] = 0
            q.append((start_x, start_y))
            
            while q:
                x, y = q.popleft()
                for dx, dy in knight_moves:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        if dist[nx][ny] == -1:
                            dist[nx][ny] = dist[x][y] + 1
                            q.append((nx, ny))
            # Update D for all target points
            for j in range(n_p):
                tx, ty = points[j]
                D[i][j] = dist[tx][ty]
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(current_pos, mask):
            if mask == 0:
                return 0
            remaining = bin(mask).count('1')
            moves_made = n - remaining
            is_alice_turn = (moves_made % 2 == 0)
            
            if is_alice_turn:
                max_total = -float('inf')
                for j in range(1, n + 1):
                    if mask & (1 << (j - 1)):
                        steps = D[current_pos][j]
                        new_mask = mask ^ (1 << (j - 1))
                        total = steps + dp(j, new_mask)
                        if total > max_total:
                            max_total = total
                return max_total
            else:
                min_total = float('inf')
                for j in range(1, n + 1):
                    if mask & (1 << (j - 1)):
                        steps = D[current_pos][j]
                        new_mask = mask ^ (1 << (j - 1))
                        total = steps + dp(j, new_mask)
                        if total < min_total:
                            min_total = total
                return min_total
        
        initial_mask = (1 << n) - 1  # All pawns are present initially
        return dp(0, initial_mask)