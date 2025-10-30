class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are exactly 4 possible 2x2 squares in a 3x3 grid:
        # top-left corners at (0,0), (0,1), (1,0), (1,1).

        def can_form_square_2x2(cells: List[str]) -> bool:
            # Check if we can have all 'B' by changing at most one cell:
            mismatches_for_B = sum(1 for c in cells if c == 'W')
            if mismatches_for_B <= 1:
                return True

            # Check if we can have all 'W' by changing at most one cell:
            mismatches_for_W = sum(1 for c in cells if c == 'B')
            if mismatches_for_W <= 1:
                return True

            return False

        # List all the top-left corners of possible 2x2 squares
        possible_squares = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # For each 2x2 square, collect the 4 cells and check
        for r, c in possible_squares:
            cells = [
                grid[r][c],
                grid[r][c + 1],
                grid[r + 1][c],
                grid[r + 1][c + 1]
            ]
            if can_form_square_2x2(cells):
                return True

        return False