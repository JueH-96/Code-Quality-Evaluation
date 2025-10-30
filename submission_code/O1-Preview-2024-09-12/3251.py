from typing import List
class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = -1
        max_area = -1
        for length, width in dimensions:
            diag = (length**2 + width**2)**0.5
            area = length * width
            if diag > max_diagonal:
                max_diagonal = diag
                max_area = area
            elif diag == max_diagonal:
                if area > max_area:
                    max_area = area
        return max_area