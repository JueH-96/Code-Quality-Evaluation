from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Precompute the minimum number of moves for the knight to reach each pawn
        def bfs(start_x, start_y, targets):
            dist = {}
            q = deque()
            q.append((start_x, start_y))
            dist[(start_x, start_y)] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in [(-2, -1), (-1, -2), (1, -2), (2, -1),
                               (2, 1), (1, 2), (-1, 2), (-2, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in dist:
                        dist[(nx, ny)] = dist[(x, y)] + 1
                        q.append((nx, ny))
            # Get the distances for the targets
            res = {}
            for tx, ty in targets:
                res[(tx, ty)] = dist.get((tx, ty), float('inf'))
            return res
        
        # Convert positions to tuples for easier handling
        pawns = [tuple(pos) for pos in positions]
        # Compute the initial distances from the knight to all pawns
        initial_dist = bfs(kx, ky, pawns)
        # Now, we need to simulate the game
        # Alice and Bob take turns, Alice wants to maximize the total moves, Bob wants to minimize
        # We can model this as a game where Alice and Bob choose pawns in a way that maximizes or minimizes the sum
        # Since the order of choosing pawns affects the total moves, we need to consider all possible sequences
        # Given the small number of pawns (up to 15), we can use dynamic programming with bitmasking
        # dp[mask][player] represents the maximum sum of moves when the remaining pawns are represented by mask and it's player's turn
        # player 0 is Alice, player 1 is Bob
        n = len(pawns)
        # Precompute the distance between every pair of pawns
        dist_pawns = {}
        for i in range(n):
            dist_pawns[i] = bfs(pawns[i][0], pawns[i][1], pawns)
        # Initialize the DP table
        # mask is a bitmask representing which pawns are still on the board
        # We will use a dictionary to store the DP values
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(mask, player, current_pos):
            if mask == 0:
                return 0
            if player == 0:
                # Alice's turn, maximize the sum
                max_sum = -float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        # Calculate the distance from current_pos to pawns[i]
                        if current_pos == (kx, ky):
                            d = initial_dist[pawns[i]]
                        else:
                            d = dist_pawns[current_pos][pawns[i]]
                        new_mask = mask ^ (1 << i)
                        sum_moves = d + dp(new_mask, 1, pawns[i])
                        if sum_moves > max_sum:
                            max_sum = sum_moves
                return max_sum
            else:
                # Bob's turn, minimize the sum
                min_sum = float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        # Calculate the distance from current_pos to pawns[i]
                        if current_pos == (kx, ky):
                            d = initial_dist[pawns[i]]
                        else:
                            d = dist_pawns[current_pos][pawns[i]]
                        new_mask = mask ^ (1 << i)
                        sum_moves = d + dp(new_mask, 0, pawns[i])
                        if sum_moves < min_sum:
                            min_sum = sum_moves
                return min_sum
        # Start with all pawns on the board, Alice's turn, and the knight at (kx, ky)
        initial_mask = (1 << n) - 1
        return dp(initial_mask, 0, (kx, ky))