class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = 0
        max_area = 0
        
        for length, width in dimensions:
            diagonal_sq = length*length + width*width
            area = length * width
            
            # If we find a rectangle with a larger diagonal, update max_diagonal_sq and max_area
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = area
            # If diagonal is the same, take the rectangle with the larger area
            elif diagonal_sq == max_diagonal_sq and area > max_area:
                max_area = area
        
        return max_area