from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # There are at most 15 pawns. Let node 0 be the starting knight position
        # and nodes 1..n be the pawn positions.
        n = len(positions)
        coords = [(kx, ky)] + [tuple(pos) for pos in positions]
        
        # Precompute knight distance between any two of these positions.
        # We'll use BFS on a 50x50 board.
        N = 50  # board size
        
        # Moves of knight.
        deltas = [(-2, -1), (-1, -2), (1, -2), (2, -1),
                  (2, 1), (1, 2), (-1, 2), (-2, 1)]
        
        # Cache for distances between nodes.
        dist = [[0]*(n+1) for _ in range(n+1)]
        
        # For each starting position among our key nodes, do BFS.
        def bfs(start_x, start_y):
            # returns a 2D array with distance from (start_x, start_y) to each cell.
            grid = [[-1]*N for _ in range(N)]
            q = deque()
            q.append((start_x, start_y))
            grid[start_x][start_y] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in deltas:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and grid[nx][ny] == -1:
                        grid[nx][ny] = grid[x][y] + 1
                        q.append((nx, ny))
            return grid
        
        # Precompute distance maps for each of our nodes.
        dist_maps = []
        for x, y in coords:
            dist_maps.append(bfs(x,y))
        
        # Fill the dist table for each pair of nodes i, j.
        for i in range(n+1):
            for j in range(n+1):
                # Coordinates for node j:
                xj, yj = coords[j]
                d = dist_maps[i][xj][yj]
                if d == -1:
                    # Though a knight in open board should be able to reach anywhere 
                    # given enough moves.
                    d = 10**9
                dist[i][j] = d

        # We use DP with bitmask over the captured pawns. We have (n) pawns.
        # State: (mask, pos, turn) where:
        #   mask: bitmask of pawns that have been captured; 0 <= mask < (1 << n)
        #   pos: the index in coords where knight currently is (0 for start, 1..n for a pawn captured last)
        #   turn: boolean, True means it's Alice's turn (maximizing), False for Bob (minimizing).
        from functools import lru_cache
        ALL = (1 << n) - 1
        
        @lru_cache(maxsize=None)
        def dp(mask: int, pos: int, aliceTurn: bool) -> int:
            if mask == ALL:
                return 0
            # Variables to track result based on turn.
            if aliceTurn:
                best = -10**9  # maximize
                for i in range(n):
                    if not (mask & (1 << i)):
                        # Note: pawn i corresponds to node i+1.
                        moveCost = dist[pos][i+1]
                        # Proceed to new state: mark pawn as captured, new pos is i+1, turn toggles.
                        next_val = dp(mask | (1 << i), i+1, not aliceTurn)
                        best = max(best, moveCost + next_val)
                return best
            else:
                worst = 10**9  # minimize
                for i in range(n):
                    if not (mask & (1 << i)):
                        moveCost = dist[pos][i+1]
                        next_val = dp(mask | (1 << i), i+1, not aliceTurn)
                        worst = min(worst, moveCost + next_val)
                return worst

        return dp(0, 0, True)