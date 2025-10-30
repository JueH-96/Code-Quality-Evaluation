import math

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = 0
        max_area = 0
        for dimension in dimensions:
            length = dimension[0]
            width = dimension[1]
            current_diagonal_sq = length * length + width * width
            current_area = length * width
            if current_diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = current_diagonal_sq
                max_area = current_area
            elif current_diagonal_sq == max_diagonal_sq:
                max_area = max(max_area, current_area)
        return max_area