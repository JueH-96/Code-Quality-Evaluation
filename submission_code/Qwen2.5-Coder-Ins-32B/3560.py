from typing import List
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        directions = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        def bfs(start_x, start_y, target_x, target_y):
            queue = [(start_x, start_y, 0)]
            visited = set()
            visited.add((start_x, start_y))
            
            while queue:
                x, y, moves = queue.pop(0)
                if x == target_x and y == target_y:
                    return moves
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')
        
        @lru_cache(None)
        def dp(kx, ky, pawns):
            if not pawns:
                return 0
            max_moves = 0
            for i in range(len(pawns)):
                px, py = pawns[i]
                moves_to_pawn = bfs(kx, ky, px, py)
                remaining_pawns = pawns[:i] + pawns[i+1:]
                max_moves = max(max_moves, moves_to_pawn + dp(px, py, remaining_pawns))
            return max_moves
        
        return dp(kx, ky, tuple(map(tuple, positions)))