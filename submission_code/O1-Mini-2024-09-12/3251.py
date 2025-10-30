class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag_sq = -1
        max_area = -1
        for length, width in dimensions:
            diag_sq = length ** 2 + width ** 2
            area = length * width
            if diag_sq > max_diag_sq or (diag_sq == max_diag_sq and area > max_area):
                max_diag_sq = diag_sq
                max_area = area
        return max_area