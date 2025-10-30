from typing import List, Tuple
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Directions a knight can move in chess
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]

        def bfs(start: Tuple[int, int], target: Tuple[int, int]) -> int:
            queue = deque([(start[0], start[1], 0)])
            visited = set()
            visited.add((start[0], start[1]))

            while queue:
                x, y, moves = queue.popleft()
                if (x, y) == target:
                    return moves
                for dx, dy in knight_moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')

        total_moves = 0
        current_position = (kx, ky)

        while positions:
            # Alice's turn
            min_moves = float('inf')
            best_pawn = None
            for px, py in positions:
                moves = bfs(current_position, (px, py))
                if moves < min_moves:
                    min_moves = moves
                    best_pawn = (px, py)
            total_moves += min_moves
            current_position = best_pawn
            positions.remove(best_pawn)

            if not positions:
                break

            # Bob's turn
            max_moves = 0
            worst_pawn = None
            for px, py in positions:
                moves = bfs(current_position, (px, py))
                if moves > max_moves:
                    max_moves = moves
                    worst_pawn = (px, py)
            total_moves += max_moves
            current_position = worst_pawn
            positions.remove(worst_pawn)

        return total_moves