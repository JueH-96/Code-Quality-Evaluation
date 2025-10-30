class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_squared = 0
        max_area = 0
        
        for length, width in dimensions:
            # Calculate diagonal squared (to avoid sqrt calculation)
            diagonal_squared = length * length + width * width
            area = length * width
            
            # If we found a longer diagonal, update both diagonal and area
            if diagonal_squared > max_diagonal_squared:
                max_diagonal_squared = diagonal_squared
                max_area = area
            # If diagonal is the same as max, choose the one with larger area
            elif diagonal_squared == max_diagonal_squared:
                max_area = max(max_area, area)
        
        return max_area