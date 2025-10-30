from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Number of pawns
        n = len(positions)
        # We will treat the knight's starting square as an extra "source" at index n
        sources = positions + [(kx, ky)]
        # All 8 possible knight moves
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2),  (1, 2),  (2, -1),  (2, 1)]
        
        # Precompute shortest‐path distances on the empty 50x50 board
        # from each source (each pawn and the start) to each pawn.
        # dist[s][j] = min knight‐moves from sources[s] to positions[j]
        dist = [[0] * n for _ in range(n + 1)]
        for s in range(n + 1):
            sx, sy = sources[s]
            # BFS grid
            dmap = [[-1] * 50 for _ in range(50)]
            dq = deque()
            dmap[sx][sy] = 0
            dq.append((sx, sy))
            while dq:
                x, y = dq.popleft()
                d0 = dmap[x][y]
                for dx, dy in knight_moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dmap[nx][ny] < 0:
                        dmap[nx][ny] = d0 + 1
                        dq.append((nx, ny))
            # Record distances to each pawn
            for j in range(n):
                tx, ty = positions[j]
                dist[s][j] = dmap[tx][ty]
        
        # We will do a minimax dynamic programming over subsets of pawns.
        # mask is a bitmask of remaining pawns. 1 means the pawn is still there.
        # pos ∈ [0..n] is the current knight position index (n means the start).
        full_mask = (1 << n) - 1
        dp = [[-1] * (n + 1) for _ in range(1 << n)]
        bit_count = int.bit_count
        
        def dfs(mask: int, pos: int) -> int:
            # No pawns left
            if mask == 0:
                return 0
            if dp[mask][pos] != -1:
                return dp[mask][pos]
            
            # How many have been captured so far?
            captured = n - bit_count(mask)
            # Alice moves on even captured counts (0,2,4,...), Bob on odd.
            is_alice = (captured % 2 == 0)
            
            if is_alice:
                # Alice tries to maximize total moves
                best = 0
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    cost = dist[pos][j] + dfs(mask ^ lsb, j)
                    if cost > best:
                        best = cost
                    m ^= lsb
            else:
                # Bob tries to minimize total moves
                best = 10**9
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    cost = dist[pos][j] + dfs(mask ^ lsb, j)
                    if cost < best:
                        best = cost
                    m ^= lsb
            
            dp[mask][pos] = best
            return best
        
        # Start with all pawns present and the knight at index n (the start)
        return dfs(full_mask, n)