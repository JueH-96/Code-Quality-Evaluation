from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def knight_moves(x, y):
            moves = []
            for dx, dy in [(-2, -1), (-1, -2), (1, -2), (2, -1), 
                           (2, 1), (1, 2), (-1, 2), (-2, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50:
                    moves.append((nx, ny))
            return moves
        
        def bfs(start, target):
            queue = deque([start])
            visited = set()
            visited.add(start)
            distance = 0
            
            while queue:
                for _ in range(len(queue)):
                    x, y = queue.popleft()
                    if (x, y) == target:
                        return distance
                    for nx, ny in knight_moves(x, y):
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                distance += 1
            return float('inf')
        
        n = len(positions)
        dp = [[-1] * (1 << n) for _ in range(n + 1)]
        
        def dfs(turn, mask):
            if mask == (1 << n) - 1:
                return 0
            
            if dp[turn][mask] != -1:
                return dp[turn][mask]
            
            best = float('-inf') if turn == 0 else float('inf')
            
            for i in range(n):
                if mask & (1 << i) == 0:
                    moves_needed = bfs((kx, ky), tuple(positions[i]))
                    new_mask = mask | (1 << i)
                    next_turn = 1 - turn
                    result = moves_needed + dfs(next_turn, new_mask)
                    
                    if turn == 0:
                        best = max(best, result)
                    else:
                        best = min(best, result)
            
            dp[turn][mask] = best
            return best
        
        return dfs(0, 0)