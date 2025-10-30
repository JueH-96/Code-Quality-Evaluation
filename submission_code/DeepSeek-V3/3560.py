from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Precompute the minimum number of moves for the knight to reach each pawn
        def bfs(start_x, start_y):
            dist = [[-1 for _ in range(50)] for _ in range(50)]
            q = deque()
            q.append((start_x, start_y))
            dist[start_x][start_y] = 0
            moves = [(-2, -1), (-1, -2), (1, -2), (2, -1),
                     (2, 1), (1, 2), (-1, 2), (-2, 1)]
            while q:
                x, y = q.popleft()
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dist[nx][ny] == -1:
                        dist[nx][ny] = dist[x][y] + 1
                        q.append((nx, ny))
            return dist
        
        # Get the distances from the knight's starting position
        dist = bfs(kx, ky)
        
        # Extract the distances for the pawns
        pawn_distances = []
        for x, y in positions:
            pawn_distances.append(dist[x][y])
        
        # Sort the distances in descending order
        pawn_distances.sort(reverse=True)
        
        # Alice and Bob take turns picking the largest remaining distance
        total_moves = 0
        for i, d in enumerate(pawn_distances):
            if i % 2 == 0:
                total_moves += d
        return total_moves