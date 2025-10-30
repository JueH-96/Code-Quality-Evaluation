from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def bfs(start_x, start_y, end_x, end_y):
            directions = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            queue = deque([(start_x, start_y, 0)])
            visited = set((start_x, start_y))
            
            while queue:
                x, y, moves = queue.popleft()
                if x == end_x and y == end_y:
                    return moves
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')
        
        def minimax(kx, ky, positions, is_alice_turn):
            if not positions:
                return 0
            
            if is_alice_turn:
                max_moves = 0
                for i, (px, py) in enumerate(positions):
                    moves = bfs(kx, ky, px, py)
                    remaining_positions = positions[:i] + positions[i+1:]
                    max_moves = max(max_moves, moves + minimax(px, py, remaining_positions, not is_alice_turn))
                return max_moves
            else:
                min_moves = float('inf')
                for i, (px, py) in enumerate(positions):
                    moves = bfs(kx, ky, px, py)
                    remaining_positions = positions[:i] + positions[i+1:]
                    min_moves = min(min_moves, moves + minimax(px, py, remaining_positions, not is_alice_turn))
                return min_moves
        
        return minimax(kx, ky, positions, True)