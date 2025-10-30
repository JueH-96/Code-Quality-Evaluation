from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        used = set()
        max_score = 0

        def backtrack(row: int, score: int, used: set) -> None:
            nonlocal max_score
            if row == m:
                max_score = max(max_score, score)
                return
            for col in range(n):
                if grid[row][col] not in used:
                    used.add(grid[row][col])
                    backtrack(row + 1, score + grid[row][col], used)
                    used.remove(grid[row][col])

        backtrack(0, 0, used)
        return max_score