class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are exactly four 2x2 sub-squares in a 3x3 grid:
        # top-left, top-right, bottom-left, bottom-right.
        # For each 2x2 square, count how many 'B' and 'W' there are.
        # If there are 4 of the same color, or 3 of one color and 1 of the other,
        # or 1 of one color and 3 of the other, we can form a 2x2 square by flipping at most one cell.

        # Define the coordinates for the top-left corners of each 2x2 sub-square
        sub_squares = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # Check each sub-square
        for r, c in sub_squares:
            # Extract the cells of the 2x2 sub-square
            cells = [
                grid[r][c],     grid[r][c+1],
                grid[r+1][c],   grid[r+1][c+1]
            ]
            countB = cells.count('B')
            countW = 4 - countB  # or cells.count('W')

            # If we have all same color or can get it by flipping one cell
            # Cases: (4,0), (3,1), (1,3), (0,4)
            if countB == 4 or countB == 3 or countB == 1 or countB == 0:
                return True
        
        return False