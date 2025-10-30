from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        cols = list(range(n))
        res, best = 0, 0
        def dfs(path, index):
            nonlocal res
            if index == n:
                score = sum(grid[path[i-1]][col] for i, col in enumerate(path) if i > 0)
                res = max(res, score)
                return
            for col in cols[index:]:
                path.append(col)
                cols[index] = cols[index].index(col)
                dfs(path, index + 1)
                path.pop()
        dfs([], 0)
        return res