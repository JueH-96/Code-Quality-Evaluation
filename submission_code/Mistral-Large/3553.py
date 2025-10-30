class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Function to determine the color of a given square
        def get_color(coordinate):
            # Calculate the sum of the column and row indices
            col = ord(coordinate[0]) - ord('a') + 1
            row = int(coordinate[1])
            # If the sum is even, the square is white; otherwise, it's black
            return (col + row) % 2 == 0

        # Check if both squares have the same color
        return get_color(coordinate1) == get_color(coordinate2)