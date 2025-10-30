from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        INF = 10**9
        moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
        n = len(positions)
        points = [(kx, ky)] + positions
        
        dist_matrix = [[0] * (n+1) for _ in range(n+1)]
        
        for i in range(n+1):
            dist_grid = [[-1] * 50 for _ in range(50)]
            sx, sy = points[i]
            dist_grid[sx][sy] = 0
            q = deque()
            q.append((sx, sy))
            while q:
                x, y = q.popleft()
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and dist_grid[nx][ny] == -1:
                        dist_grid[nx][ny] = dist_grid[x][y] + 1
                        q.append((nx, ny))
            for j in range(n+1):
                xj, yj = points[j]
                d_val = dist_grid[xj][yj]
                if d_val == -1:
                    d_val = INF
                dist_matrix[i][j] = d_val
        
        memo = {}
        
        def dp(mask, pos_index):
            if mask == 0:
                return 0
            key = (mask, pos_index)
            if key in memo:
                return memo[key]
            
            cnt = bin(mask).count("1")
            turn = (n - cnt) % 2
            
            res = -10**18 if turn == 0 else 10**18
            for i in range(n):
                if mask & (1 << i):
                    cost = dist_matrix[pos_index][i+1]
                    if cost >= INF:
                        continue
                    new_mask = mask ^ (1 << i)
                    next_moves = dp(new_mask, i+1)
                    total = cost + next_moves
                    if turn == 0:
                        if total > res:
                            res = total
                    else:
                        if total < res:
                            res = total
            if res == -10**18 or res == 10**18:
                res = 0
            memo[key] = res
            return res
        
        return dp((1 << n) - 1, 0)