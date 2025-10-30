import math
from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = 0
        max_area = 0
        
        for length, width in dimensions:
            # Calculate the diagonal length using Pythagorean theorem
            diagonal = math.sqrt(length**2 + width**2)
            # Calculate the area of the rectangle
            area = length * width
            
            # Check if this rectangle has a longer diagonal
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = area
            # If diagonals are the same, choose the rectangle with the larger area
            elif diagonal == max_diagonal:
                max_area = max(max_area, area)
        
        return max_area