class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        rows, cols = len(grid), len(grid[0])
        result = []
        count = 0  # Position counter in zigzag sequence
        
        for row in range(rows):
            if row % 2 == 0:  # Even row, go right
                for col in range(cols):
                    if count % 2 == 0:  # Include only even positions
                        result.append(grid[row][col])
                    count += 1
            else:  # Odd row, go left
                for col in range(cols - 1, -1, -1):
                    if count % 2 == 0:  # Include only even positions
                        result.append(grid[row][col])
                    count += 1
        
        return result