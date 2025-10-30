class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Helper function to determine the color of a given chessboard square.
        # A square's color can be determined by the parity of the sum of its column and row indices.
        # For the chessboard, we can map columns 'a' to 'h' to 1 to 8.
        def square_color(coordinate: str) -> bool:
            col = ord(coordinate[0]) - ord('a') + 1
            row = int(coordinate[1])
            # if (col + row) is even, we assign one color; if odd, the opposite color.
            return (col + row) % 2 == 0
        
        # Return True if both coordinates have the same color, False otherwise.
        return square_color(coordinate1) == square_color(coordinate2)