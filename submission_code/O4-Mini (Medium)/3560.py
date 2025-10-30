from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Number of pawns
        n = len(positions)
        # Special points: index 0 is initial knight, 1..n are pawns
        coords = [(kx, ky)] + [(x, y) for x, y in positions]
        m = n + 1  # total special points
        
        # Precompute knight distances between all special points
        # dist_mat[p][q] = min knight moves from coords[p] to coords[q]
        dist_mat = [[0] * m for _ in range(m)]
        # Knight moves
        moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
        # Board dimensions
        N = 50
        
        for p in range(m):
            # BFS from coords[p]
            dist_grid = [[-1]*N for _ in range(N)]
            sx, sy = coords[p]
            dq = deque()
            dq.append((sx, sy))
            dist_grid[sx][sy] = 0
            while dq:
                x, y = dq.popleft()
                d0 = dist_grid[x][y]
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and dist_grid[nx][ny] < 0:
                        dist_grid[nx][ny] = d0 + 1
                        dq.append((nx, ny))
            # Fill dist_mat[p][q]
            for q in range(m):
                tx, ty = coords[q]
                dist_mat[p][q] = dist_grid[tx][ty]
        
        # Precompute parity: parity[mask] = True if it's Alice's turn (even bits captured)
        full_mask = (1 << n) - 1
        parity = [True] * (1 << n)
        for mask in range(1, 1 << n):
            # remove lowest set bit flips parity
            parity[mask] = not parity[mask & (mask - 1)]
        
        # DP table: dp[mask][p] = optimal total moves from state with captured=mask, knight at p
        # mask bits 0..n-1 correspond to pawns indices 0..n-1 (special index = i+1)
        dp = [[0] * m for _ in range(1 << n)]
        
        # Bottom-up over masks
        # When mask == full_mask, dp[mask][p] = 0 (no pawns left)
        # Fill for mask from full_mask-1 down to 0
        for mask in range(full_mask - 1, -1, -1):
            isAlice = parity[mask]
            for p in range(m):
                if isAlice:
                    best = -10**9
                else:
                    best = 10**18
                # Try capturing each remaining pawn i
                for i in range(n):
                    bit = 1 << i
                    if (mask & bit) == 0:
                        # cost to move from p to pawn i+1, plus dp after capturing
                        cost = dist_mat[p][i+1] + dp[mask | bit][i+1]
                        if isAlice:
                            if cost > best:
                                best = cost
                        else:
                            if cost < best:
                                best = cost
                dp[mask][p] = best
        
        # Answer starting with no pawns captured, knight at initial (index 0)
        return dp[0][0]