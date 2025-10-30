class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Determine the color of a square based on its coordinates
        def get_color(coordinate: str) -> str:
            col = coordinate[0]
            row = int(coordinate[1])
            # Chessboard coloring pattern: (col + row) % 2
            # If the sum is even, the square is black; if odd, the square is white
            return 'black' if (ord(col) - ord('a') + row) % 2 == 0 else 'white'

        # Get the colors of the two coordinates
        color1 = get_color(coordinate1)
        color2 = get_color(coordinate2)

        # Return True if both squares have the same color, False otherwise
        return color1 == color2