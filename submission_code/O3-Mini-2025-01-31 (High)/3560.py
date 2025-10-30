from collections import deque
from functools import lru_cache
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # The idea is to precompute the knight distance between the starting position
        # and every pawn, and between every pair of pawns.
        # Then we use a minimax DP over the remaining pawns.
        # Let positions_list[0] be the knight's starting position,
        # and positions_list[1:] be the pawn positions.
        # In each turn, the current player must choose one pawn (say at index j)
        # and incur a cost = (minimum knight moves from current position to pawn j)
        # and update the knight's current position to that pawn.
        # Alice (turn=0) wants to maximize the total cost,
        # while Bob (turn=1) wants to minimize it.
        
        # Build the list of important positions: start + all pawn positions.
        pts = [(kx, ky)] + [ (x, y) for x, y in positions ]
        n = len(positions)  # number of pawns
        
        # Precompute the knight moves distances between these positions.
        # The board is 50 x 50.
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
        total_positions = n + 1
        cost = [[0] * total_positions for _ in range(total_positions)]
        
        for i in range(total_positions):
            sx, sy = pts[i]
            # BFS to compute distances from (sx, sy) to all squares on board
            dist = [[-1] * 50 for _ in range(50)]
            dq = deque()
            dq.append((sx, sy))
            dist[sx][sy] = 0
            while dq:
                x, y = dq.popleft()
                for dx, dy in knight_moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dist[nx][ny] == -1:
                        dist[nx][ny] = dist[x][y] + 1
                        dq.append((nx, ny))
            # Fill distances for other important positions.
            for j in range(total_positions):
                tx, ty = pts[j]
                cost[i][j] = dist[tx][ty]
                
        # Now use DP with minimax (Alice maximizes, Bob minimizes).
        # We use a bitmask to represent the set of remaining pawns.
        # Pawns are stored at indices 1..n in pts, so we use bits 0..(n-1)
        # with bit i corresponding to pawn at index i+1.
        @lru_cache(maxsize=None)
        def dp(mask: int, cur: int, turn: int) -> int:
            # mask: bitmask of remaining pawns, cur: current index in pts, turn: 0 for Alice, 1 for Bob.
            if mask == 0:
                return 0
            if turn == 0:  # Alice maximizes the total moves.
                best = -float('inf')
                # iterate over each pawn available (bit i in mask)
                for i in range(n):
                    if mask & (1 << i):
                        next_idx = i + 1  # pawn's index in pts
                        new_mask = mask ^ (1 << i)
                        # Move to pawn at next_idx incurs cost[cur][next_idx]
                        total_moves = cost[cur][next_idx] + dp(new_mask, next_idx, 1)
                        best = max(best, total_moves)
                return best
            else:  # Bob minimizes the total moves.
                best = float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        next_idx = i + 1
                        new_mask = mask ^ (1 << i)
                        total_moves = cost[cur][next_idx] + dp(new_mask, next_idx, 0)
                        best = min(best, total_moves)
                return best
        
        initial_mask = (1 << n) - 1
        return dp(initial_mask, 0, 0)