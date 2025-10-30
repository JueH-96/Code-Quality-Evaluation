class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Define the four 2x2 squares by their cell positions
        squares = [
            [(0,0), (0,1), (1,0), (1,1)],  # Top-left
            [(0,1), (0,2), (1,1), (1,2)],  # Top-right
            [(1,0), (1,1), (2,0), (2,1)],  # Bottom-left
            [(1,1), (1,2), (2,1), (2,2)]   # Bottom-right
        ]
        
        for square in squares:
            cells = [grid[x][y] for x, y in square]
            count_b = cells.count('B')
            count_w = cells.count('W')
            
            # If all four cells are the same
            if count_b == 4 or count_w == 4:
                return True
            # If three cells are the same and one is different
            if count_b == 3 or count_w == 3:
                return True
        
        # If no square can be made uniform with at most one change
        return False