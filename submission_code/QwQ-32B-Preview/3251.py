from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_sum = -1
        max_area = -1
        
        for rect in dimensions:
            length, width = rect
            sum_sq = length**2 + width**2
            area = length * width
            
            if sum_sq > max_sum:
                max_sum = sum_sq
                max_area = area
            elif sum_sq == max_sum:
                if area > max_area:
                    max_area = area
        
        return max_area