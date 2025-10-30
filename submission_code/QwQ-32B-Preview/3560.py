from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Collect all positions: starting position + pawn positions
        all_positions = [(kx, ky)] + positions
        m = len(positions)
        num_positions = m + 1  # including starting position
        
        # Precompute distances between all pairs of positions
        distance = [[0] * num_positions for _ in range(num_positions)]
        for i in range(num_positions):
            for j in range(num_positions):
                if i != j:
                    distance[i][j] = self.bfs_distance(all_positions[i], all_positions[j])
        
        # Initialize dp table
        dp = [[0] * num_positions for _ in range(1 << m)]
        
        # Iterate over all masks from 1 to (1 << m) - 1
        for mask in range(1, 1 << m):
            k = bin(mask).count('1')
            # Determine whose turn it is
            if k % 2 == 1:  # Alice's turn
                for pos in range(num_positions):
                    dp[mask][pos] = max(
                        distance[pos][p] + dp[mask ^ (1 << p)][p]
                        for p in range(1, num_positions) if (mask & (1 << p))
                    )
            else:  # Bob's turn
                for pos in range(num_positions):
                    dp[mask][pos] = min(
                        distance[pos][p] + dp[mask ^ (1 << p)][p]
                        for p in range(1, num_positions) if (mask & (1 << p))
                    )
        
        # The answer is dp[(1 << m) - 1][0]
        return dp[(1 << m) - 1][0]
    
    def bfs_distance(self, start, end):
        if start == end:
            return 0
        queue = deque()
        visited = set()
        queue.append((start, 0))
        visited.add(start)
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        while queue:
            (x, y), steps = queue.popleft()
            for dx, dy in knight_moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                    if (nx, ny) == end:
                        return steps + 1
                    queue.append(((nx, ny), steps + 1))
                    visited.add((nx, ny))
        return -1  # Should not happen if end is reachable