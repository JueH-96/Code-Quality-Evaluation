class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_sq = -1
        max_area_for_max_diagonal = 0
        
        for length, width in dimensions:
            diagonal_sq = length*length + width*width
            area = length * width
            
            if diagonal_sq > max_diagonal_sq:
                # Found a new larger diagonal
                max_diagonal_sq = diagonal_sq
                max_area_for_max_diagonal = area
            elif diagonal_sq == max_diagonal_sq:
                # If diagonals are equal, choose the one with the larger area
                if area > max_area_for_max_diagonal:
                    max_area_for_max_diagonal = area
        
        return max_area_for_max_diagonal