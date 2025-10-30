from typing import List
from functools import lru_cache
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        """
        Minimax on the set of remaining pawns.
        State  : (mask , current_position_index , turn)
                 turn = 0 -> Alice (maximiser)
                 turn = 1 -> Bob   (minimiser)
        ------------------------------------------------------------------
        mask   : bit i is 1   -> pawn i is still on the board
        indexes: 0..n-1  -> pawns
                 n       -> current knight is on the initial square
        ------------------------------------------------------------------
        Transition
                 If it is players turn, choose one remaining pawn j
                 cost  = distance[current_pos][j] +
                         dp(mask without j , j , next_turn)
                 Alice picks the maximal cost, Bob the minimal.
        ------------------------------------------------------------------
        Distances between every pair of relevant squares
        (< 16 of them) are pre–computed with BFS on the 50×50 board.
        """
        
        # --------------- pre-compute knight distances -------------------
        moves = [( 2, 1),( 2,-1),(-2, 1),(-2,-1),
                 ( 1, 2),( 1,-2),(-1, 2),(-1,-2)]
        
        # list of all interesting squares (all pawns + knight start)
        pawns = [tuple(p) for p in positions]
        n     = len(pawns)
        start_idx = n                       # index assigned to knight start
        nodes = pawns + [(kx, ky)]          # total n+1 squares
        
        # BFS from one square to every board cell
        def bfs(src: tuple) -> List[List[int]]:
            sx, sy = src
            dist = [[-1]*50 for _ in range(50)]
            dist[sx][sy] = 0
            dq = deque([(sx, sy)])
            while dq:
                x, y = dq.popleft()
                d = dist[x][y] + 1
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dist[nx][ny] == -1:
                        dist[nx][ny] = d
                        dq.append((nx, ny))
            return dist
        
        # distance matrix between every pair of interesting squares
        dist = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            dmap = bfs(nodes[i])
            for j in range(n+1):
                x, y = nodes[j]
                dist[i][j] = dmap[x][y]
        
        # --------------------------- DP ---------------------------------
        @lru_cache(None)
        def dp(mask: int, pos_idx: int, turn: int) -> int:
            if mask == 0:                         # no pawns left
                return 0
            if turn == 0:                         # Alice (maximize)
                best = -1
                m = mask
                while m:
                    j = (m & -m).bit_length() - 1   # index of one pawn
                    cost = dist[pos_idx][j] + dp(mask ^ (1 << j), j, 1)
                    if cost > best:
                        best = cost
                    m &= m - 1
                return best
            else:                                 # Bob (minimize)
                best = 10**9
                m = mask
                while m:
                    j = (m & -m).bit_length() - 1
                    cost = dist[pos_idx][j] + dp(mask ^ (1 << j), j, 0)
                    if cost < best:
                        best = cost
                    m &= m - 1
                return best
        
        full_mask = (1 << n) - 1
        return dp(full_mask, start_idx, 0)