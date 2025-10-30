from collections import deque
from functools import lru_cache
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Number of pawns
        n = len(positions)
        # All relevant source points: 0 is the knight start, 1..n are the pawn positions
        sources = [(kx, ky)] + [(x, y) for x, y in positions]
        
        # Knight moves
        moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                 (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        # Precompute distances from each source to each pawn.
        # dists[i][j] = min knight moves from sources[i] to positions[j]
        dists = [[0]*n for _ in range(n+1)]
        
        # For each source point, do a BFS on the 50x50 board
        for i, (sx, sy) in enumerate(sources):
            dist_grid = [[-1]*50 for _ in range(50)]
            dq = deque([(sx, sy)])
            dist_grid[sx][sy] = 0
            while dq:
                x, y = dq.popleft()
                d = dist_grid[x][y]
                for dx, dy in moves:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dist_grid[nx][ny] == -1:
                        dist_grid[nx][ny] = d + 1
                        dq.append((nx, ny))
            # extract distances to each pawn
            for j, (px, py) in enumerate(positions):
                dists[i][j] = dist_grid[px][py]
        
        # dp(mask, last_src, turn) -> best total from here
        # mask: bitmask of captured pawns
        # last_src: index in sources for knight's current pos (0..n)
        # turn: 0 for Alice (max), 1 for Bob (min)
        @lru_cache(None)
        def dp(mask: int, last_src: int, turn: int) -> int:
            if mask == (1 << n) - 1:
                return 0
            if turn == 0:
                # Alice's turn: maximize
                best = 0
                for j in range(n):
                    if not (mask & (1 << j)):
                        cost = dists[last_src][j]
                        total = cost + dp(mask | (1 << j), j+1, 1)
                        if total > best:
                            best = total
                return best
            else:
                # Bob's turn: minimize
                best = float('inf')
                for j in range(n):
                    if not (mask & (1 << j)):
                        cost = dists[last_src][j]
                        total = cost + dp(mask | (1 << j), j+1, 0)
                        if total < best:
                            best = total
                return best
        
        # Start with no pawns captured, knight at source 0, Alice's turn (0)
        return dp(0, 0, 0)