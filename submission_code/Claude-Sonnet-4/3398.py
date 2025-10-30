class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 squares in the 3x3 grid
        squares = [
            [(0,0), (0,1), (1,0), (1,1)],  # top-left
            [(0,1), (0,2), (1,1), (1,2)],  # top-right
            [(1,0), (1,1), (2,0), (2,1)],  # bottom-left
            [(1,1), (1,2), (2,1), (2,2)]   # bottom-right
        ]
        
        for square in squares:
            # Count colors in this 2x2 square
            colors = []
            for r, c in square:
                colors.append(grid[r][c])
            
            # Count B's and W's
            b_count = colors.count('B')
            w_count = colors.count('W')
            
            # If all same color (4-0) or 3-1 split, we can make it uniform
            if b_count == 4 or w_count == 4 or b_count >= 3 or w_count >= 3:
                return True
        
        return False