from collections import deque
from typing import List, Tuple
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Board dimensions
        N = 50
        
        # Knight moves: (dx, dy)
        moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        # We'll store nodes: starting position (kx, ky) and each pawn position.
        nodes = [(kx, ky)]
        for pos in positions:
            nodes.append((pos[0], pos[1]))
        total_nodes = len(nodes)  # 1 + number of pawns
        
        # Precompute minimal knight moves between each pair of nodes using BFS.
        # dist[i][j] will be minimal moves from nodes[i] to nodes[j]
        dist = [[0] * total_nodes for _ in range(total_nodes)]
        
        def bfs(start: Tuple[int, int]) -> List[List[int]]:
            # distances on board from starting cell "start"
            dgrid = [[-1] * N for _ in range(N)]
            sx, sy = start
            dgrid[sx][sy] = 0
            q = deque()
            q.append((sx, sy))
            while q:
                x, y = q.popleft()
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and dgrid[nx][ny] == -1:
                        dgrid[nx][ny] = dgrid[x][y] + 1
                        q.append((nx, ny))
            return dgrid
        
        # For each node compute BFS distances.
        bfs_results = []
        for i in range(total_nodes):
            bfs_results.append(bfs(nodes[i]))
        
        # Fill in dist matrix for every pair of nodes.
        for i in range(total_nodes):
            for j in range(total_nodes):
                x, y = nodes[j]
                dist[i][j] = bfs_results[i][x][y]
        
        # There are n pawns, indices 1..n in nodes list.
        n = total_nodes - 1
        
        # We'll use dp with bitmask state.
        # We define dp(cur, mask, turn) where:
        # cur: current index in nodes list where the knight is (0 for starting, 1..n for pawn positions already captured)
        # mask: bitmask for available pawns (bit i corresponds to pawn at node i+1 is still available)
        # turn: 0 means Alice's turn (maximizer), 1 means Bob's turn (minimizer)
        @lru_cache(maxsize=None)
        def dp(cur: int, mask: int, turn: int) -> int:
            # Base case: no pawns remaining.
            if mask == 0:
                return 0
            # If it's Alice's turn, she tries to maximize the sum.
            if turn == 0:
                best = -10**9
                # Iterate over all available pawns.
                # Pawn index i corresponds to node index = i+1.
                m = mask
                i = 0
                while m:
                    # get the lowest set bit:
                    if m & 1:
                        next_node = i + 1
                        cost = dist[cur][next_node]
                        # Continue game with updated state and turn flipped.
                        value = cost + dp(next_node, mask & ~(1 << i), 1)
                        if value > best:
                            best = value
                    m //= 2
                    i += 1
                return best
            else:
                # Bob's turn. He minimizes the eventual total.
                worst = 10**9
                m = mask
                i = 0
                while m:
                    if m & 1:
                        next_node = i + 1
                        cost = dist[cur][next_node]
                        value = cost + dp(next_node, mask & ~(1 << i), 0)
                        if value < worst:
                            worst = value
                    m //= 2
                    i += 1
                return worst
        
        # Initially, knight is at node index 0 (starting position), all pawns available, and it's Alice's turn.
        full_mask = (1 << n) - 1
        return dp(0, full_mask, 0)