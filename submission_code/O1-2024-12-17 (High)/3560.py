from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        """
        We use a Minimax DP (with bitmask) to find the maximum total number of moves
        that Alice can force if both players play optimally (Alice maximizing, Bob minimizing).

        Steps:
        1) Precompute knight distances among all relevant "positions":
           - Each of the pawn locations
           - The knight's initial location
           We will have p = len(positions) pawns, plus 1 extra index for the knight's start
           (total of p+1 "slots").
           dist[i][j] will store the minimum knight moves from slot i to slot j.

        2) Define a DP with state (pos, mask, turn), where
           - pos is the current slot of the knight (0..p-1 for pawns, p for the initial knight position).
           - mask is a bitmask indicating which pawns remain on the board (1 means it remains).
           - turn is 0 for Alice, 1 for Bob (Alice tries to maximize total moves, Bob tries to minimize).
           
           Let dp[turn][pos][mask] be the optimal result (total moves from now on) in that state.

           If mask == 0, return 0  (no more pawns to capture).
           Otherwise:
             if turn == 0 (Alice's turn):
               dp[0][pos][mask] = max( dist[pos][pawn] + dp[1][pawn][mask without that pawn] )
                 over all pawns still in mask
             if turn == 1 (Bob's turn):
               dp[1][pos][mask] = min( dist[pos][pawn] + dp[0][pawn][mask without that pawn] )
                 over all pawns still in mask

        3) The answer is dp[0][p][(1<<p) - 1], meaning:
           - Knight is initially at index p (the "start" index),
           - All pawns are present in the mask ((1<<p)-1),
           - It's Alice's turn (turn=0).
        """

        # All possible knight moves
        KNIGHT_MOVES = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
        
        # Board limits
        N = 50
        
        # 1) Build the list of all relevant locations: p pawn-positions + 1 start-position
        p = len(positions)
        locs = positions[:]  # copy
        locs.append([kx, ky])  # index p is the knight's starting position
        
        # 2) Precompute distances dist[i][j]: # of knight moves from locs[i] to locs[j]
        # We'll do a BFS from each locs[i].
        dist = [[0]*(p+1) for _ in range(p+1)]  # (p+1) x (p+1)
        
        def bfs(startx, starty):
            """ Returns a 2D array of knight-distances from (startx, starty) to all squares. """
            dist2d = [[-1]*N for _ in range(N)]
            dist2d[startx][starty] = 0
            queue = deque()
            queue.append((startx, starty))
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in KNIGHT_MOVES:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < N and 0 <= ny < N and dist2d[nx][ny] == -1:
                        dist2d[nx][ny] = dist2d[cx][cy] + 1
                        queue.append((nx, ny))
            return dist2d
        
        # Compute BFS for each of the p+1 slots
        bfs_maps = []
        for i in range(p+1):
            x0, y0 = locs[i]
            bfs_maps.append(bfs(x0, y0))
        
        # Fill dist[i][j] using bfs_maps
        for i in range(p+1):
            for j in range(p+1):
                if i != j:
                    xj, yj = locs[j]
                    dist[i][j] = bfs_maps[i][xj][yj]
                else:
                    dist[i][j] = 0  # Not really used
                
        # 3) Set up a DP array: dp[turn][pos][mask]
        # turn in {0,1}, pos in [0..p], mask in [0..(1<<p)-1]
        # We'll use -1 to denote uninitialized.
        MAX_MASK = 1 << p
        dp = [[[-1]*(MAX_MASK) for _ in range(p+1)] for _ in range(2)]
        
        def solve(pos, mask, turn):
            """Return the optimal total (max or min) from state (pos, mask, turn)."""
            if mask == 0:
                return 0
            if dp[turn][pos][mask] != -1:
                return dp[turn][pos][mask]
            
            if turn == 0:  # Alice's turn, maximize
                best = -1
                # Try capturing each pawn in 'mask'
                m = mask
                while m > 0:
                    # pick the lowest set bit
                    pawn_idx = (m & -m).bit_length() - 1
                    # cost of capturing that pawn
                    cost = dist[pos][pawn_idx] + solve(pawn_idx, mask ^ (1 << pawn_idx), 1)
                    if cost > best:
                        best = cost
                    # clear that bit
                    m = m & (m - 1)
                dp[turn][pos][mask] = best
            else:  # Bob's turn, minimize
                best = 10**15
                m = mask
                while m > 0:
                    pawn_idx = (m & -m).bit_length() - 1
                    cost = dist[pos][pawn_idx] + solve(pawn_idx, mask ^ (1 << pawn_idx), 0)
                    if cost < best:
                        best = cost
                    m = m & (m - 1)
                dp[turn][pos][mask] = best
            
            return dp[turn][pos][mask]
        
        # Initial state: knight at index p (the start), mask = (1<<p) - 1 (all pawns present),
        # turn = 0 (Alice)
        full_mask = (1 << p) - 1
        return solve(p, full_mask, 0)