class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = 0
        max_area = 0
        
        for length, width in dimensions:
            diagonal_sq = length * length + width * width
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = length * width
            elif diagonal_sq == max_diagonal_sq:
                max_area = max(max_area, length * width)
        
        return max_area