from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag2 = -1
        max_area = 0
        
        for length, width in dimensions:
            diag2 = length * length + width * width  # Calculate the squared diagonal to avoid float issues.
            area = length * width
            
            if diag2 > max_diag2:
                max_diag2 = diag2
                max_area = area
            elif diag2 == max_diag2:
                max_area = max(max_area, area)
                
        return max_area