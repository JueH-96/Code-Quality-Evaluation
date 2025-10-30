import math
from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        M = len(positions)  # Number of pawns
        N = M + 1  # Total points including knight start
        # Create position list with tuples for easier handling
        pos_list = [(kx, ky)]
        for pos in positions:
            pos_list.append(tuple(pos))
        
        INF = 100000  # Large number to represent infinity for unreachable positions
        
        # Compute distance matrix using knight moves
        dist = [[0 for _ in range(N)] for _ in range(N)]
        
        def knight_dist(x1, y1, x2, y2):
            if x1 == x2 and y1 == y2:
                return 0
            directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            queue = deque([(x1, y1, 0)])  # (x, y, distance)
            visited = set([(x1, y1)])
            while queue:
                cx, cy, cd = queue.popleft()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx <= 49 and 0 <= ny <= 49 and (nx, ny) not in visited:
                        if nx == x2 and ny == y2:
                            return cd + 1
                        queue.append((nx, ny, cd + 1))
                        visited.add((nx, ny))
            return INF  # Unreachable
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist[i][j] = knight_dist(pos_list[i][0], pos_list[i][1], pos_list[j][0], pos_list[j][1])
        
        # Memoization table for DP
        memo = [[-1 for _ in range(N)] for _ in range(1 << M)]
        
        def dp_func(mask, pos_idx):
            if mask == 0:
                return 0  # No pawns left
            if memo[mask][pos_idx] != -1:
                return memo[mask][pos_idx]
            
            num_uncaptured = bin(mask).count('1')
            num_captured = M - num_uncaptured
            
            # Find possible pawns to capture (uncaptured and reachable)
            possible_k = []
            for k in range(M):  # k is the index in the mask (0 to M-1)
                if (mask & (1 << k)) != 0 and dist[pos_idx][k + 1] < INF:
                    possible_k.append(k)
            
            if not possible_k:
                # Cannot capture any more pawns
                memo[mask][pos_idx] = 0
                return 0
            
            if num_captured % 2 == 0:  # Even number captured, Alice's turn (maximizer)
                best = -1  # Initialize to a value less than minimum possible sum (0)
                for k in possible_k:
                    pawn_pos_idx = k + 1  # Position index in pos_list
                    dist_cost = dist[pos_idx][pawn_pos_idx]
                    new_mask = mask & ~(1 << k)
                    new_sum = dist_cost + dp_func(new_mask, pawn_pos_idx)
                    if new_sum > best:
                        best = new_sum
                memo[mask][pos_idx] = best
                return best
            else:  # Odd number captured, Bob's turn (minimizer)
                best = 2000000  # Large number, larger than maximum possible sum
                for k in possible_k:
                    pawn_pos_idx = k + 1
                    dist_cost = dist[pos_idx][pawn_pos_idx]
                    new_mask = mask & ~(1 << k)
                    new_sum = dist_cost + dp_func(new_mask, pawn_pos_idx)
                    if new_sum < best:
                        best = new_sum
                memo[mask][pos_idx] = best
                return best
        
        # Initial state: all pawns uncaptured, knight at start position (index 0)
        initial_mask = (1 << M) - 1
        return dp_func(initial_mask, 0)