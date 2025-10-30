class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Define the four possible 2x2 squares by their top-left positions
        squares = [
            [(0,0), (0,1), (1,0), (1,1)],  # Top-left square
            [(0,1), (0,2), (1,1), (1,2)],  # Top-right square
            [(1,0), (1,1), (2,0), (2,1)],  # Bottom-left square
            [(1,1), (1,2), (2,1), (2,2)]   # Bottom-right square
        ]
        
        for square in squares:
            cells = [grid[r][c] for r, c in square]
            count_B = cells.count('B')
            count_W = cells.count('W')
            
            # If all four cells are the same
            if count_B == 4 or count_W == 4:
                return True
            # If three cells are the same and one is different
            elif count_B == 3 or count_W == 3:
                return True
            # If two cells are 'B' and two are 'W', two changes are needed
            # which is more than allowed, so skip this square
        
        # If no square can be made uniform with at most one change
        return False