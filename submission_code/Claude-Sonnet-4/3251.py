class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = 0
        max_area = 0
        
        for length, width in dimensions:
            # Calculate diagonal squared to avoid floating point operations
            diagonal_sq = length * length + width * width
            area = length * width
            
            # If this diagonal is longer, update both max_diagonal_sq and max_area
            if diagonal_sq > max_diagonal_sq:
                max_diagonal_sq = diagonal_sq
                max_area = area
            # If diagonal is same but area is larger, update max_area
            elif diagonal_sq == max_diagonal_sq and area > max_area:
                max_area = area
        
        return max_area