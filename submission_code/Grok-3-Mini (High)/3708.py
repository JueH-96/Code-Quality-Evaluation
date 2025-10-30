class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        rows = len(grid)
        cols = len(grid[0])
        result = []
        count = 0
        for row in range(rows):
            if row % 2 == 0:  # even row, traverse left to right
                for col in range(cols):
                    if count % 2 == 0:
                        result.append(grid[row][col])
                    count += 1
            else:  # odd row, traverse right to left
                for col in range(cols - 1, -1, -1):
                    if count % 2 == 0:
                        result.append(grid[row][col])
                    count += 1
        return result