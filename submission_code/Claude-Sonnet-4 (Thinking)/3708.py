class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        result = []
        should_visit = True  # Start by visiting the first cell
        
        for row in range(m):
            if row % 2 == 0:  # Even row: left to right
                for col in range(n):
                    if should_visit:
                        result.append(grid[row][col])
                    should_visit = not should_visit
            else:  # Odd row: right to left
                for col in range(n-1, -1, -1):
                    if should_visit:
                        result.append(grid[row][col])
                    should_visit = not should_visit
        
        return result