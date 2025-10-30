from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = float('-inf')
        max_area = 0
        for rect in dimensions:
            l, w = rect[0], rect[1]
            diagonal_sq = l ** 2 + w ** 2
            area = l * w
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = area
            elif diagonal_sq == max_diagonal_sq:
                if area > max_area:
                    max_area = area
        return max_area