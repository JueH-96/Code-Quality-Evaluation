class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_squared_diagonal = 0
        max_area = 0
        
        for l, w in dimensions:
            squared_diagonal = l**2 + w**2
            area = l * w
            
            if squared_diagonal > max_squared_diagonal or (squared_diagonal == max_squared_diagonal and area > max_area):
                max_squared_diagonal = squared_diagonal
                max_area = area
        
        return max_area