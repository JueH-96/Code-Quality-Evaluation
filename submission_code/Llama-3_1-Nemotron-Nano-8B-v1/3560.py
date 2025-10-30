from typing import List
from collections import deque

class Solution:
    def __init__(self):
        self.dx = [2, 1, -1, -2, -2, -1, 1, 2]
        self.dy = [1, 2, 2, 1, -1, -2, -2, -1]
        self.dist = [[[[ -1 for _ in range(50)] for __ in range(50)] for ___ in range(50)] for ____ in range(50)]
        
        for kx in range(50):
            for ky in range(50):
                visited = [[-1 for _ in range(50)] for __ in range(50)]
                queue = deque()
                queue.append((kx, ky))
                visited[kx][ky] = 0
                while queue:
                    x, y = queue.popleft()
                    for i in range(8):
                        nx = x + self.dx[i]
                        ny = y + self.dy[i]
                        if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                            visited[nx][ny] = visited[x][y] + 1
                            queue.append((nx, ny))
                for px in range(50):
                    for py in range(50):
                        self.dist[kx][ky][px][py] = visited[px][py] if visited[px][py] != -1 else -1

    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        pos_list = positions
        indexed_positions = [(x, y, i) for i, (x, y) in enumerate(pos_list)]
        initial_mask = (1 << n) - 1
        memo = {}
        
        def dfs(current_kx, current_ky, mask, is_alice_turn):
            if mask == 0:
                return 0
            key = (current_kx, current_ky, mask, is_alice_turn)
            if key in memo:
                return memo[key]
            
            pawn_indices = [i for i in range(n) if (mask & (1 << i))]
            best = -float('inf') if is_alice_turn else float('inf')
            
            for i in pawn_indices:
                px, py, _ = indexed_positions[i]
                steps = self.dist[current_kx][current_ky][px][py]
                if steps == -1:
                    continue
                new_mask = mask & ~(1 << i)
                res = steps + dfs(px, py, new_mask, not is_alice_turn)
                if is_alice_turn:
                    if res > best:
                        best = res
                else:
                    if res < best:
                        best = res
            
            if best == -float('inf'):
                memo[key] = 0
                return 0
            memo[key] = best
            return best
        
        return dfs(kx, ky, initial_mask, True)