class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag_sq = -1
        max_area = 0
        for rect in dimensions:
            l, w = rect
            diag_sq = l ** 2 + w ** 2
            area = l * w
            if diag_sq > max_diag_sq:
                max_diag_sq = diag_sq
                max_area = area
            elif diag_sq == max_diag_sq:
                if area > max_area:
                    max_area = area
        return max_area