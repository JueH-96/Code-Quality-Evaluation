class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        if not grid or not grid[0]:
            return []
        
        rows = len(grid)
        cols = len(grid[0])
        result = []
        
        for row in range(rows):
            # For even rows, traverse left to right
            if row % 2 == 0:
                # Start from first column
                for col in range(0, cols, 2):
                    result.append(grid[row][col])
            # For odd rows, traverse right to left
            else:
                # Start from last column if cols is odd, or second last if cols is even
                start = cols - 1 if cols % 2 == 1 else cols - 2
                for col in range(start, -1, -2):
                    result.append(grid[row][col])
        
        return result