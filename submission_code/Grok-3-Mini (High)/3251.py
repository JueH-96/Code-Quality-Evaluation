class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag_sq = 0
        max_area = 0
        for length, width in dimensions:
            diag_sq = length ** 2 + width ** 2
            area = length * width
            if diag_sq > max_diag_sq:
                max_diag_sq = diag_sq
                max_area = area
            elif diag_sq == max_diag_sq:
                if area > max_area:
                    max_area = area
        return max_area