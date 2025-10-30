from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Possible knight moves
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        # Function to calculate the minimum moves from (sx, sy) to (tx, ty)
        def min_moves(sx, sy, tx, ty):
            if (sx, sy) == (tx, ty):
                return 0
            visited = set()
            queue = deque([(sx, sy, 0)])  # (current_x, current_y, current_moves)
            visited.add((sx, sy))
            
            while queue:
                x, y, moves = queue.popleft()
                for dx, dy in knight_moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        if (nx, ny) == (tx, ty):
                            return moves + 1
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')  # Should never reach here if the board is valid
        
        # Calculate all pairwise distances from the knight to each pawn
        distances = [min_moves(kx, ky, px, py) for px, py in positions]
        
        # Sort distances in descending order to maximize Alice's moves
        distances.sort(reverse=True)
        
        # Alice and Bob take turns picking the maximum available move
        total_moves = 0
        for i, dist in enumerate(distances):
            if i % 2 == 0:  # Alice's turn
                total_moves += dist
            # Bob's turn does not add to total_moves as he minimizes it
        
        return total_moves