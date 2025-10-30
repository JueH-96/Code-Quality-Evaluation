from typing import List
from functools import lru_cache
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Preprocess coordinates
        coords = [(kx, ky)]
        for pos in positions:
            coords.append(tuple(pos))
        m = len(coords)
        n = len(positions)
        
        # Precompute distance matrix
        dist = [[-1] * m for _ in range(m)]
        dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for s in range(m):
            sx, sy = coords[s]
            grid = [[-1] * 50 for _ in range(50)]
            queue = deque()
            queue.append((sx, sy))
            grid[sx][sy] = 0
            while queue:
                x, y = queue.popleft()
                current_dist = grid[x][y]
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and grid[nx][ny] == -1:
                        grid[nx][ny] = current_dist + 1
                        queue.append((nx, ny))
            for t in range(m):
                tx, ty = coords[t]
                dist[s][t] = grid[tx][ty]
        
        @lru_cache(maxsize=None)
        def dp(current_pos, mask):
            if mask == 0:
                return 0
            k = bin(mask).count('1')
            # Determine if it's Alice's turn (maximizer) or Bob's (minimizer)
            is_alice_turn = (n - k) % 2 == 0
            if is_alice_turn:
                max_total = -1
                for pawn_idx in range(n):
                    if mask & (1 << pawn_idx):
                        pawn_coord_idx = pawn_idx + 1
                        steps = dist[current_pos][pawn_coord_idx]
                        new_mask = mask ^ (1 << pawn_idx)
                        total = steps + dp(pawn_coord_idx, new_mask)
                        if total > max_total:
                            max_total = total
                return max_total
            else:
                min_total = float('inf')
                for pawn_idx in range(n):
                    if mask & (1 << pawn_idx):
                        pawn_coord_idx = pawn_idx + 1
                        steps = dist[current_pos][pawn_coord_idx]
                        new_mask = mask ^ (1 << pawn_idx)
                        total = steps + dp(pawn_coord_idx, new_mask)
                        if total < min_total:
                            min_total = total
                return min_total
        
        initial_mask = (1 << n) - 1
        return dp(0, initial_mask)