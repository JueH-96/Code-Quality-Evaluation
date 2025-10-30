from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Define the possible moves of a knight
        moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

        # Function to calculate the minimum number of moves to reach a position
        def min_moves(x1, y1, x2, y2):
            queue = deque([(x1, y1, 0)])
            visited = {(x1, y1)}
            while queue:
                x, y, steps = queue.popleft()
                if (x, y) == (x2, y2):
                    return steps
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        queue.append((nx, ny, steps + 1))
                        visited.add((nx, ny))

        # Calculate the minimum number of moves to reach each pawn
        moves_to_pawns = [min_moves(kx, ky, x, y) for x, y in positions]

        # Sort the pawns by the minimum number of moves to reach them
        sorted_pawns = sorted(zip(moves_to_pawns, positions))

        # Initialize the total number of moves
        total_moves = 0

        # Simulate the game
        for i, (moves_to_pawn, _) in enumerate(sorted_pawns):
            # If it's Alice's turn, add the minimum number of moves to reach the pawn
            if i % 2 == 0:
                total_moves += moves_to_pawn
            # If it's Bob's turn, add the minimum number of moves to reach the pawn
            else:
                total_moves += moves_to_pawn

        return total_moves