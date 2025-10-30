class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = 0
        max_area = 0
        
        for length, width in dimensions:
            # Calculate the square of the diagonal to avoid floating point operations
            diagonal_sq = length * length + width * width
            area = length * width
            
            # If this rectangle has a longer diagonal, update both max_diagonal_sq and max_area
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = area
            # If the diagonal is the same but area is larger, update max_area
            elif diagonal_sq == max_diagonal_sq:
                max_area = max(max_area, area)
        
        return max_area