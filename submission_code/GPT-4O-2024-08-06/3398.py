class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 squares in the grid
        for i in range(2):
            for j in range(2):
                # Extract the 2x2 square
                square = [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]]
                # Count the number of 'B's and 'W's
                count_B = square.count('B')
                count_W = square.count('W')
                # If there are 3 of one color and 1 of the other, we can change the 1 to make a 2x2 square
                if count_B == 3 or count_W == 3:
                    return True
                # If there are 4 of one color, it's already a 2x2 square
                if count_B == 4 or count_W == 4:
                    return True
        # If no such 2x2 square can be formed, return False
        return False