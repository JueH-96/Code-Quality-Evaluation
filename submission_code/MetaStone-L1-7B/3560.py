from collections import deque
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        min_steps = [0] * n

        for i in range(n):
            x, y = positions[i]
            visited = [[-1 for _ in range(50)] for _ in range(50)]
            q = deque()
            q.append((kx, ky))
            visited[kx][ky] = 0
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                               (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        if visited[nx][ny] == -1:
                            visited[nx][ny] = visited[cx][cy] + 1
                            q.append((nx, ny))
                            if nx == x and ny == y:
                                min_steps[i] = visited[nx][ny]
                                break  # Found the minimal steps for this pawn
            # The BFS might have broken before reaching the target, but in our problem statement, all pawns are reachable
            # So, min_steps should have the correct minimal steps

        @lru_cache(maxsize=None)
        def game(mask, is_alice_turn):
            if mask == 0:
                return 0
            if is_alice_turn:
                max_total = 0
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue  # Skip if pawn is already captured
                    move = min_steps[i]
                    new_mask = mask & ~ (1 << i)
                    total = move + game(new_mask, False)
                    if total > max_total:
                        max_total = total
                return max_total
            else:
                min_total = float('inf')
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue  # Skip if pawn is already captured
                    move = min_steps[i]
                    new_mask = mask & ~ (1 << i)
                    total = move + game(new_mask, True)
                    if total < min_total:
                        min_total = total
                return min_total

        initial_mask = (1 << n) - 1
        return game(initial_mask, True)