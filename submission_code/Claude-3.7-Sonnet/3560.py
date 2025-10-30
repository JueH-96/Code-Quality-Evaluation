from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Calculate minimum moves for knight to go from (x1,y1) to (x2,y2)
        def min_knight_moves(x1, y1, x2, y2, cache):
            if (x1, y1, x2, y2) in cache:
                return cache[(x1, y1, x2, y2)]
            
            # Knight's possible moves
            moves = [
                (1, 2), (2, 1), (2, -1), (1, -2),
                (-1, -2), (-2, -1), (-2, 1), (-1, 2)
            ]
            
            # BFS to find shortest path
            queue = deque([(x1, y1, 0)])  # (x, y, steps)
            visited = {(x1, y1)}
            
            while queue:
                x, y, steps = queue.popleft()
                
                if x == x2 and y == y2:
                    cache[(x1, y1, x2, y2)] = steps
                    return steps
                
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx <= 49 and 0 <= ny <= 49 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, steps + 1))
            
            return -1  # Unreachable (shouldn't happen in this problem)
        
        # Minimax algorithm with memoization
        def minimax(knight_x, knight_y, pawns, is_alice, memo, move_cache):
            if not pawns:
                return 0  # No more pawns to capture
            
            # Check if we've already computed this state
            state = (knight_x, knight_y, pawns, is_alice)
            if state in memo:
                return memo[state]
            
            if is_alice:
                # Alice's turn - maximize total moves
                best_score = float('-inf')
                for i in range(len(pawns)):
                    px, py = pawns[i]
                    # Calculate moves to capture this pawn
                    moves = min_knight_moves(knight_x, knight_y, px, py, move_cache)
                    # Remove the captured pawn
                    new_pawns = tuple(pawns[j] for j in range(len(pawns)) if j != i)
                    # Recursive call for Bob's turn
                    next_score = minimax(px, py, new_pawns, False, memo, move_cache)
                    # Update best score
                    best_score = max(best_score, moves + next_score)
            else:
                # Bob's turn - minimize total moves
                best_score = float('inf')
                for i in range(len(pawns)):
                    px, py = pawns[i]
                    # Calculate moves to capture this pawn
                    moves = min_knight_moves(knight_x, knight_y, px, py, move_cache)
                    # Remove the captured pawn
                    new_pawns = tuple(pawns[j] for j in range(len(pawns)) if j != i)
                    # Recursive call for Alice's turn
                    next_score = minimax(px, py, new_pawns, True, memo, move_cache)
                    # Update best score
                    best_score = min(best_score, moves + next_score)
            
            memo[state] = best_score
            return best_score
        
        # Convert positions to tuple of tuples for hashability
        positions_tuple = tuple(tuple(p) for p in positions)
        memo = {}
        move_cache = {}
        
        return minimax(kx, ky, positions_tuple, True, memo, move_cache)