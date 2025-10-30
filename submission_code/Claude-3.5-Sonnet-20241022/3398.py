class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 squares and see if we can make them same color
        # with at most one change
        
        def check_square(r, c):
            # Get colors in current 2x2 square
            colors = [
                grid[r][c], grid[r][c+1],
                grid[r+1][c], grid[r+1][c+1]
            ]
            
            # Count occurrences of each color
            b_count = colors.count('B')
            w_count = colors.count('W')
            
            # If all same color already, return True
            if b_count == 4 or w_count == 4:
                return True
                
            # If we can make all black with one change
            if b_count == 3 and w_count == 1:
                return True
                
            # If we can make all white with one change    
            if w_count == 3 and b_count == 1:
                return True
                
            return False
            
        # Check all possible 2x2 squares
        for i in range(2):
            for j in range(2):
                if check_square(i, j):
                    return True
                    
        return False