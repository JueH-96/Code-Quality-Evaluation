from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = -1
        max_area = -1
        
        for rect in dimensions:
            l, w = rect[0], rect[1]
            diagonal_sq = l * l + w * w
            area = l * w
            
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = area
            elif diagonal_sq == max_diagonal_sq:
                if area > max_area:
                    max_area = area
                    
        return max_area