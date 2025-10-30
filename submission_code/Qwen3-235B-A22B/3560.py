from typing import List
from collections import deque
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n_pawns = len(positions)
        if n_pawns == 0:
            return 0
        
        # Prepare all positions (knight's initial position and all pawns)
        all_positions = [(kx, ky)] + [(x, y) for x, y in positions]
        n_total = n_pawns + 1  # number of all positions (knight + pawns)
        
        # Precompute distance matrix: dist[i][j] is distance from all_positions[i] to pawn j's position
        dist = [[0] * n_pawns for _ in range(n_total)]
        
        # Directions a knight can move
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        
        # For each position in all_positions, run BFS to compute distances to all pawns
        for i in range(n_total):
            x_start, y_start = all_positions[i]
            visited = [[-1] * 50 for _ in range(50)]
            q = deque()
            visited[x_start][y_start] = 0
            q.append((x_start, y_start))
            
            # BFS to compute minimal steps to all cells
            while q:
                x, y = q.popleft()
                for dx, dy in knight_moves:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        q.append((nx, ny))
            
            # Extract distances to each pawn's position
            for j in range(n_pawns):
                px, py = positions[j]
                dist[i][j] = visited[px][py]
        
        # Dynamic programming with memoization
        @lru_cache(maxsize=None)
        def dp(mask, pos_idx):
            if mask == 0:
                return 0
            
            mask_count = bin(mask).count('1')
            captured = n_pawns - mask_count
            is_alice = (captured % 2 == 0)
            
            best = 0 if is_alice else float('inf')
            
            for j in range(n_pawns):
                if mask & (1 << j):
                    d = dist[pos_idx][j]
                    new_mask = mask ^ (1 << j)
                    new_pos_idx = j + 1  # all_positions[j+1] is pawn j's position
                    
                    sub_result = dp(new_mask, new_pos_idx)
                    current_total = d + sub_result
                    
                    if is_alice:
                        if current_total > best:
                            best = current_total
                    else:
                        if current_total < best:
                            best = current_total
            
            return best
        
        initial_mask = (1 << n_pawns) - 1
        initial_pos_idx = 0
        return dp(initial_mask, initial_pos_idx)