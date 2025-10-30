class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        import math
        max_diagonal = 0
        max_area = 0
        
        for length, width in dimensions:
            diagonal = math.sqrt(length**2 + width**2)
            area = length * width
            
            if diagonal > max_diagonal:
                max_diagonal = diagonal
                max_area = area
            elif diagonal == max_diagonal:
                if area > max_area:
                    max_area = area
        
        return max_area