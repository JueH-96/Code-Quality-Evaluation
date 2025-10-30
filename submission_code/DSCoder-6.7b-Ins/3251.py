import math

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = 0
        max_area = 0
        for i in range(len(dimensions)):
            diagonal = math.sqrt(dimensions[i][0]**2 + dimensions[i][1]**2)
            area = dimensions[i][0] * dimensions[i][1]
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = area
            elif diagonal == max_diagonal:
                max_area = max(max_area, area)
        return max_area