class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 squares in the 3x3 grid
        for i in range(2):  # Only need to check rows 0 and 1
            for j in range(2):  # Only need to check columns 0 and 1
                # Count the number of 'B' and 'W' in the 2x2 square
                count_B = sum(grid[i+x][j+y] == 'B' for x in range(2) for y in range(2))
                count_W = 4 - count_B  # Since there are 4 cells in total
                
                # If we have 4 of the same color or 3 of one color and 1 of the other
                if count_B == 4 or count_W == 4 or count_B == 3 or count_W == 3:
                    return True
        
        return False