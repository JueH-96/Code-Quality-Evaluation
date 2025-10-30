class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        skip = False
        
        for row in range(len(grid)):
            if row % 2 == 0:  # Even rows: left to right
                for col in range(len(grid[row])):
                    if not skip:
                        result.append(grid[row][col])
                    skip = not skip
            else:  # Odd rows: right to left
                for col in range(len(grid[row]) - 1, -1, -1):
                    if not skip:
                        result.append(grid[row][col])
                    skip = not skip
        
        return result