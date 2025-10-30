from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag_sq = 0
        max_area = 0
        for dim in dimensions:
            l, w = dim[0], dim[1]
            current_diag_sq = l ** 2 + w ** 2
            current_area = l * w
            if current_diag_sq > max_diag_sq:
                max_diag_sq = current_diag_sq
                max_area = current_area
            elif current_diag_sq == max_diag_sq:
                if current_area > max_area:
                    max_area = current_area
        return max_area