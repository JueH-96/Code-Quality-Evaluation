from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        result = []
        
        # Flag to determine if the current cell should be visited
        # Starts True because the first cell (0,0) is always visited
        should_visit_next = True 
        
        for i in range(m):
            if i % 2 == 0:  # Even row: Traverse left to right
                for j in range(n):
                    if should_visit_next:
                        result.append(grid[i][j])
                    should_visit_next = not should_visit_next # Toggle flag for the next cell
            else:  # Odd row: Traverse right to left
                for j in range(n - 1, -1, -1):
                    if should_visit_next:
                        result.append(grid[i][j])
                    should_visit_next = not should_visit_next # Toggle flag for the next cell
                    
        return result