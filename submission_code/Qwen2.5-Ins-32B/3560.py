from typing import List
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        @lru_cache(None)
        def min_moves(x1, y1, x2, y2):
            if x1 == x2 and y1 == y2:
                return 0
            moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            queue = [(x1, y1, 0)]
            visited = set([(x1, y1)])
            while queue:
                x, y, steps = queue.pop(0)
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        if nx == x2 and ny == y2:
                            return steps + 1
                        visited.add((nx, ny))
                        queue.append((nx, ny, steps + 1))
            return float('inf')

        @lru_cache(None)
        def dfs(kx, ky, state):
            if state == 0:
                return 0
            moves = []
            for i in range(len(positions)):
                if state & (1 << i):
                    x, y = positions[i]
                    next_state = state ^ (1 << i)
                    moves.append(min_moves(kx, ky, x, y) + dfs(x, y, next_state))
            return max(moves) if state.bit_count() % 2 == 1 else min(moves)

        initial_state = (1 << len(positions)) - 1
        return dfs(kx, ky, initial_state)