from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        res = []
        for i in range(len(grid)):
            row = grid[i]
            if i % 2 == 0:
                res.extend(row)
            else:
                res.extend(row[::-1])
        return res[::2]