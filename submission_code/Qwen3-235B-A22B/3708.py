from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        traversal = []
        for i in range(len(grid)):
            if i % 2 == 0:
                traversal.extend(grid[i])
            else:
                traversal.extend(grid[i][::-1])
        return traversal[::2]