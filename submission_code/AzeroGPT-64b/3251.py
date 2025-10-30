from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        # Calculate area and diagonal for each rectangle, then
        # return the max area among the ones with the longest diagonal
        max_diagonal = -1
        area = -1

        for rect in dimensions:
            diagonal = rect[0] ** 2 + rect[1] ** 2
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                area = rect[0] * rect[1]
            elif diagonal == max_diagonal:
                area = max(area, rect[0] * rect[1])  # If same diagonal, keep the bigger rect

        return area