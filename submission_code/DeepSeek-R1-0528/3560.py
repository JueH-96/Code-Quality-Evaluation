from collections import deque
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        if n == 0:
            return 0
        nodes = [(kx, ky)]
        nodes.extend(positions)
        INF = 10**9
        dist = [[INF] * (n + 1) for _ in range(n + 1)]
        dirs = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        
        for i in range(n + 1):
            vis = [[-1] * 50 for _ in range(50)]
            x0, y0 = nodes[i]
            dq = deque()
            dq.append((x0, y0))
            vis[x0][y0] = 0
            while dq:
                x, y = dq.popleft()
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and vis[nx][ny] == -1:
                        vis[nx][ny] = vis[x][y] + 1
                        dq.append((nx, ny))
            for j in range(n + 1):
                xj, yj = nodes[j]
                if vis[xj][yj] == -1:
                    dist[i][j] = INF
                else:
                    dist[i][j] = vis[xj][yj]
        
        masks_by_count = [[] for _ in range(n + 1)]
        total_masks = 1 << n
        for mask in range(total_masks):
            cnt = mask.bit_count()
            masks_by_count[cnt].append(mask)
        
        dp = [[0] * total_masks for _ in range(n + 1)]
        
        for r in range(1, n + 1):
            for mask in masks_by_count[r]:
                turn = (n - r) % 2
                for cur in range(n + 1):
                    best = -10**18 if turn == 0 else 10**18
                    found = False
                    for j in range(n):
                        if mask & (1 << j):
                            if dist[cur][j + 1] >= INF:
                                continue
                            new_mask = mask ^ (1 << j)
                            total_val = dist[cur][j + 1] + dp[j + 1][new_mask]
                            found = True
                            if turn == 0:
                                if total_val > best:
                                    best = total_val
                            else:
                                if total_val < best:
                                    best = total_val
                    if found:
                        dp[cur][mask] = best
        return dp[0][total_masks - 1]