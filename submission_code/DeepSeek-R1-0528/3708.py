class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        path = []
        for i in range(len(grid)):
            if i % 2 == 0:
                path.extend(grid[i])
            else:
                path.extend(grid[i][::-1])
        return path[::2]