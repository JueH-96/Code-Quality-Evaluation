from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag_sq = dimensions[0][0] ** 2 + dimensions[0][1] ** 2
        max_area = dimensions[0][0] * dimensions[0][1]
        
        for rect in dimensions[1:]:
            l, w = rect[0], rect[1]
            diag_sq = l ** 2 + w ** 2
            area = l * w
            if diag_sq > max_diag_sq:
                max_diag_sq = diag_sq
                max_area = area
            elif diag_sq == max_diag_sq:
                if area > max_area:
                    max_area = area
        return max_area