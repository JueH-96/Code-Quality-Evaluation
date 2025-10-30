class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Function to determine the color of a square
        def is_black(coordinate: str) -> bool:
            # Convert the letter to a number (a=1, b=2, ..., h=8)
            column = ord(coordinate[0]) - ord('a') + 1
            # Convert the row to a number
            row = int(coordinate[1])
            # A square is black if the sum of its column and row is odd
            return (column + row) % 2 == 1
        
        # Check if both coordinates have the same color
        return is_black(coordinate1) == is_black(coordinate2)