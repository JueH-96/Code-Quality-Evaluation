class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # A standard chessboard has alternating colors.
        # The color of a square depends on the parity (even or odd) of the sum
        # of its row and column indices.
        # We can map columns 'a' through 'h' to numbers (e.g., 0 through 7)
        # and rows '1' through '8' to numbers (e.g., 0 through 7).
        # The color is determined by the parity of the sum of these indices.
        #
        # For example, using 0-indexed columns ('a'->0, ..., 'h'->7)
        # and 0-indexed rows ('1'->0, ..., '8'->7):
        # The color corresponds to (col_index + row_index) % 2.
        #
        # An equivalent and simpler way to get a consistent color parity value
        # (0 for one color, 1 for the other) is to use the ASCII value of the
        # column character and the integer value of the row character.
        # The parity of the sum (ord(col) + int(row)) is the same as the parity
        # of the sum of the 0-indexed indices (ord(col) - ord('a') + int(row) - 1)
        # because they differ by a constant even number (ord('a') + 1 = 97 + 1 = 98).
        # Two numbers that differ by an even number have the same parity.

        # Calculate the color parity for coordinate1 using the sum of
        # the ASCII value of the column and the integer value of the row.
        # The result modulo 2 gives a 0 or 1 representing the color type.
        color1_parity = (ord(coordinate1[0]) + int(coordinate1[1])) % 2

        # Calculate the color parity for coordinate2 using the same method.
        color2_parity = (ord(coordinate2[0]) + int(coordinate2[1])) % 2

        # If the parities are the same, the two squares have the same color.
        return color1_parity == color2_parity