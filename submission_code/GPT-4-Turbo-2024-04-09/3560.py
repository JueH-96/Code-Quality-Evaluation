from typing import List
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Directions a knight can move on a chessboard
        directions = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        # Memoization for minimum moves from one point to another
        @lru_cache(None)
        def min_moves(start_x, start_y, end_x, end_y):
            if (start_x, start_y) == (end_x, end_y):
                return 0
            queue = [(start_x, start_y, 0)]
            visited = set()
            visited.add((start_x, start_y))
            
            while queue:
                x, y, moves = queue.pop(0)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        if (nx, ny) == (end_x, end_y):
                            return moves + 1
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')  # Should never reach here for valid knight moves
        
        # Dynamic programming to solve the game optimally
        @lru_cache(None)
        def dp(state, knight_x, knight_y):
            if state == 0:
                return 0
            
            moves_list = []
            for i in range(len(positions)):
                if state & (1 << i):
                    pawn_x, pawn_y = positions[i]
                    move_cost = min_moves(knight_x, knight_y, pawn_x, pawn_y)
                    new_state = state & ~(1 << i)
                    # Calculate the result of the game after this move
                    result = move_cost + dp(new_state, pawn_x, pawn_y)
                    moves_list.append(result)
            
            if len(moves_list) % 2 == 0:  # Bob's turn, he minimizes the result
                return min(moves_list)
            else:  # Alice's turn, she maximizes the result
                return max(moves_list)
        
        initial_state = (1 << len(positions)) - 1
        return dp(initial_state, kx, ky)