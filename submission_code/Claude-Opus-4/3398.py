class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all four possible 2x2 squares
        for i in range(2):
            for j in range(2):
                # Count colors in the current 2x2 square
                count_b = 0
                count_w = 0
                
                # Check all 4 cells in the 2x2 square
                for di in range(2):
                    for dj in range(2):
                        if grid[i + di][j + dj] == 'B':
                            count_b += 1
                        else:
                            count_w += 1
                
                # If we have 3 or 4 cells of the same color, we can make a uniform square
                if count_b >= 3 or count_w >= 3:
                    return True
        
        return False