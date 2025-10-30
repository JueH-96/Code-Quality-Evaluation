from collections import deque
from functools import lru_cache
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Board dimensions
        R, C = 50, 50
        
        # All possible knight moves
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        # All points: index 0 is the knight's starting position, 1..n are the pawns
        nodes = [(kx, ky)] + positions
        n = len(positions)
        
        # Precompute the minimum knight distance between any two of these points
        # dist[i][j] = knight distance from nodes[i] to nodes[j]
        dist = [[0]*(n+1) for _ in range(n+1)]
        
        def bfs(start_r, start_c):
            """Return distance from (start_r, start_c) to every square on the board."""
            # We'll store distances to all squares (50x50) in a matrix
            dist_grid = [[-1]*C for _ in range(R)]
            q = deque()
            q.append((start_r, start_c))
            dist_grid[start_r][start_c] = 0
            while q:
                r, c = q.popleft()
                for dr, dc in knight_moves:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and dist_grid[nr][nc] == -1:
                        dist_grid[nr][nc] = dist_grid[r][c] + 1
                        q.append((nr, nc))
            return dist_grid
        
        # Run BFS from each node to get distance to all squares, then record dist to other nodes
        bfs_results = []
        for i in range(n+1):
            r, c = nodes[i]
            bfs_results.append(bfs(r, c))
        
        for i in range(n+1):
            for j in range(n+1):
                if i != j:
                    rj, cj = nodes[j]
                    dist[i][j] = bfs_results[i][rj][cj]
        
        # dp(mask, pos, turn):
        #   mask is bitmask of remaining pawns (1= present, 0= absent)
        #   pos is index of current knight position (0..n)
        #   turn = 0 if Alice's turn (maximize), 1 if Bob's turn (minimize)
        # returns the optimal total moves from this state
        @lru_cache(None)
        def dp(mask, pos, turn):
            if mask == 0:
                return 0  # No more pawns
            
            if turn == 0:
                # Alice's turn: she will choose a pawn to maximize total distance
                best = -float('inf')
                # iterate over each pawn that is still present in mask
                for pawn_index in range(n):
                    if (mask & (1 << pawn_index)) != 0:
                        # distance from pos to chosen pawn
                        moves_needed = dist[pos][pawn_index+1]
                        # next state
                        next_mask = mask ^ (1 << pawn_index)
                        val = moves_needed + dp(next_mask, pawn_index+1, 1)
                        best = max(best, val)
                return best
            else:
                # Bob's turn: he will choose a pawn to minimize total distance
                best = float('inf')
                for pawn_index in range(n):
                    if (mask & (1 << pawn_index)) != 0:
                        moves_needed = dist[pos][pawn_index+1]
                        next_mask = mask ^ (1 << pawn_index)
                        val = moves_needed + dp(next_mask, pawn_index+1, 0)
                        best = min(best, val)
                return best
        
        # Initial state: mask of all pawns present, knight at position 0 (the starting pos),
        # and it's Alice's turn (turn = 0).
        full_mask = (1 << n) - 1
        return dp(full_mask, 0, 0)