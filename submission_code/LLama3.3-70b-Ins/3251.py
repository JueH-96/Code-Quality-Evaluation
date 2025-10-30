from typing import List
import math

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = 0
        max_area = 0
        
        for length, width in dimensions:
            # Calculate the diagonal length using Pythagorean theorem
            diagonal = math.sqrt(length * length + width * width)
            
            # If the current diagonal is greater than the max diagonal, update max diagonal and max area
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = length * width
            # If the current diagonal is equal to the max diagonal, update max area if the current area is greater
            elif diagonal == max_diagonal:
                max_area = max(max_area, length * width)
        
        return max_area