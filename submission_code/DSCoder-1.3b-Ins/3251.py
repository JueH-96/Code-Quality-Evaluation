class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal = float('-inf')
        max_area = 0
        for i in range(len(dimensions)):
            diagonal = (dimensions[i][0]**2 + dimensions[i][1]**2)**0.5
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = dimensions[i][0] * dimensions[i][1]
        return max_area