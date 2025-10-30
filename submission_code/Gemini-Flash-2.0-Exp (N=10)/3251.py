import math

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = 0
        max_area = 0
        for dim in dimensions:
            length = dim[0]
            width = dim[1]
            diagonal = math.sqrt(length * length + width * width)
            area = length * width
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = area
            elif diagonal == max_diagonal:
                max_area = max(max_area, area)
        return max_area