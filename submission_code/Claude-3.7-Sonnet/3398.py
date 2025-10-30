class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Define all possible 2x2 squares in a 3x3 grid
        squares = [
            [(0,0), (0,1), (1,0), (1,1)],  # top-left
            [(0,1), (0,2), (1,1), (1,2)],  # top-right
            [(1,0), (1,1), (2,0), (2,1)],  # bottom-left
            [(1,1), (1,2), (2,1), (2,2)]   # bottom-right
        ]
        
        for square in squares:
            # Count number of black cells in this 2x2 square
            b_count = sum(1 for i, j in square if grid[i][j] == 'B')
            
            # If all cells are the same color (b_count = 0 or 4)
            # OR if we can change exactly one cell to make them all same (b_count = 1 or 3)
            if b_count in [0, 1, 3, 4]:
                return True
                
        return False