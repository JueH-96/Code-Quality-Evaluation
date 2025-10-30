class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Function to convert coordinate to numerical indices
        def get_color(coordinate):
            col = ord(coordinate[0]) - ord('a') + 1  # Converts 'a'..'h' to 1..8
            row = int(coordinate[1])                # Converts '1'..'8' to int
            # Determine color by adding column and row
            return (col + row) % 2

        # Compare the parity of the sum for both coordinates
        return get_color(coordinate1) == get_color(coordinate2)