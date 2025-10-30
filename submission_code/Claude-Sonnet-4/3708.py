class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        m, n = len(grid), len(grid[0])
        position = 0  # Track position in zigzag traversal
        
        for row in range(m):
            if row % 2 == 0:  # Even row: traverse left to right
                for col in range(n):
                    if position % 2 == 0:  # Take every alternate cell
                        result.append(grid[row][col])
                    position += 1
            else:  # Odd row: traverse right to left
                for col in range(n-1, -1, -1):
                    if position % 2 == 0:  # Take every alternate cell
                        result.append(grid[row][col])
                    position += 1
        
        return result