from typing import List
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        @lru_cache(None)
        def min_max_moves(i, j):
            if i == 0:
                return 0
            min_moves = float('inf')
            for x, y in positions:
                if (x, y) == positions[j]:
                    continue
                moves = abs(x - kx) // 2 + abs(y - ky) // 2 + (abs(x - kx) % 2 != abs(y - ky) % 2)
                if moves > 0:
                    min_moves = min(min_moves, moves + min_max_moves(i - 1, j))
            return min_moves if min_moves != float('inf') else 0
        
        n = len(positions)
        max_moves = 0
        for j in range(n):
            max_moves = max(max_moves, min_max_moves(n, j))
        return max_moves