from typing import List
from collections import defaultdict

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Dictionary to hold the diagonals
        bottom_left_diags = defaultdict(list)
        top_right_diags = defaultdict(list)
        
        # Fill the diagonals
        for i in range(n):
            for j in range(n):
                if i + j < n:  # Top-right triangle
                    top_right_diags[j - i].append(grid[i][j])
                else:  # Bottom-left triangle
                    bottom_left_diags[i - j].append(grid[i][j])
        
        # Sort the diagonals
        for key in bottom_left_diags:
            bottom_left_diags[key].sort(reverse=True)  # Non-increasing order
        
        for key in top_right_diags:
            top_right_diags[key].sort()  # Non-decreasing order
        
        # Create a new grid to fill sorted values
        result = [[0] * n for _ in range(n)]
        
        # Fill the result grid with sorted values
        for i in range(n):
            for j in range(n):
                if i + j < n:  # Top-right triangle
                    result[i][j] = top_right_diags[j - i].pop(0)
                else:  # Bottom-left triangle
                    result[i][j] = bottom_left_diags[i - j].pop(0)
        
        return result