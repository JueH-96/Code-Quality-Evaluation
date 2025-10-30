class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        
        # Helper function to determine the "color parity" of a single square.
        # It calculates (column_index + row_index) % 2.
        # Squares with the same color will have the same parity (both 0 or both 1).
        def _get_color_parity(coordinate: str) -> int:
            # The column character is the first character (e.g., 'a', 'b').
            col_char = coordinate[0]
            # The row character is the second character (e.g., '1', '2').
            row_char = coordinate[1]

            # Convert column character to a 0-indexed integer.
            # 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
            col_index = ord(col_char) - ord('a')

            # Convert row character to a 0-indexed integer.
            # '1' -> 0, '2' -> 1, ..., '8' -> 7
            row_index = int(row_char) - 1

            # The color of a square depends on the parity of the sum of its 0-indexed row and column indices.
            # If (col_index + row_index) is even, it's one color.
            # If (col_index + row_index) is odd, it's the other color.
            return (col_index + row_index) % 2

        # Calculate the color parity for the first coordinate.
        parity1 = _get_color_parity(coordinate1)
        
        # Calculate the color parity for the second coordinate.
        parity2 = _get_color_parity(coordinate2)
        
        # If the parities are equal, the squares have the same color.
        return parity1 == parity2