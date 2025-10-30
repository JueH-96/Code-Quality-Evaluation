from typing import List
from collections import deque
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        if n == 0:
            return 0
        
        # Precompute all possible starting positions (knight's initial and all pawns)
        s_positions = [(kx, ky)]
        for p in positions:
            s_positions.append((p[0], p[1]))
        
        # Define all possible knight moves
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # Precompute steps for each starting position to each pawn
        steps = {}
        for (x, y) in s_positions:
            distance = [[-1] * 50 for _ in range(50)]
            q = deque()
            q.append((x, y))
            distance[y][x] = 0
            while q:
                cx, cy = q.popleft()
                for dx, dy in knight_moves:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and distance[ny][nx] == -1:
                        distance[ny][nx] = distance[cy][cx] + 1
                        q.append((nx, ny))
            # Collect steps to each pawn
            steps_list = []
            for (px, py) in positions:
                steps_list.append(distance[py][px])
            steps[(x, y)] = steps_list
        
        @lru_cache(maxsize=None)
        def dfs(current_x: int, current_y: int, mask: int, is_alice: bool) -> int:
            if mask == 0:
                return 0
            
            current_steps_list = steps.get((current_x, current_y), [])
            if not current_steps_list:
                return 0
            
            available = []
            for i in range(n):
                if (mask & (1 << i)) and current_steps_list[i] != -1:
                    available.append(i)
            
            if not available:
                return 0
            
            if is_alice:
                max_total = -1
                for i in available:
                    new_mask = mask & ~(1 << i)
                    px, py = positions[i]
                    total = current_steps_list[i] + dfs(px, py, new_mask, False)
                    if total > max_total:
                        max_total = total
                return max_total
            else:
                min_total = float('inf')
                for i in available:
                    new_mask = mask & ~(1 << i)
                    px, py = positions[i]
                    total = current_steps_list[i] + dfs(px, py, new_mask, True)
                    if total < min_total:
                        min_total = total
                return min_total
        
        initial_mask = (1 << n) - 1
        return dfs(kx, ky, initial_mask, True)