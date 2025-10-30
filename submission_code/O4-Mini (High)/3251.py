from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        best_sq = -1
        best_area = 0
        
        for length, width in dimensions:
            diag_sq = length * length + width * width
            area = length * width
            
            # Update if we find a longer diagonal, or same diagonal but larger area
            if diag_sq > best_sq or (diag_sq == best_sq and area > best_area):
                best_sq = diag_sq
                best_area = area
        
        return best_area