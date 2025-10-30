from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag2 = -1          # largest squared diagonal found so far
        max_area   = 0          # area corresponding to the current largest diagonal
        
        for length, width in dimensions:
            diag2 = length * length + width * width   # squared diagonal
            area  = length * width                    # rectangle area
            
            if diag2 > max_diag2:                     # strictly longer diagonal
                max_diag2 = diag2
                max_area  = area
            elif diag2 == max_diag2 and area > max_area:   # same diagonal but larger area
                max_area = area
                
        return max_area