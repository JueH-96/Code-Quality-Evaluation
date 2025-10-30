import math

class Solution:
    """
    This class provides a method to determine if two chessboard squares
    have the same color.
    """
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        """
        Checks if two given chessboard squares have the same color.

        Args:
            coordinate1: A string representing the first coordinate (e.g., "a1").
            coordinate2: A string representing the second coordinate (e.g., "c3").

        Returns:
            True if the squares have the same color, False otherwise.
        """

        # Helper function to determine the 'color value' (0 or 1) of a square.
        # We can determine the color based on the parity of the sum of
        # the coordinates (when converted to 0-based indices).
        # For example, a1 -> (0, 0), sum = 0 (even) -> color 0 (e.g., black)
        # b1 -> (1, 0), sum = 1 (odd) -> color 1 (e.g., white)
        # a2 -> (0, 1), sum = 1 (odd) -> color 1 (e.g., white)
        # b2 -> (1, 1), sum = 2 (even) -> color 0 (e.g., black)
        # Squares have the same color if the parity of the sum of their indices is the same.
        def get_color_parity(coordinate: str) -> int:
            """Calculates a value (0 or 1) representing the color."""
            # Extract column character (a-h) and row character (1-8)
            col_char = coordinate[0]
            row_char = coordinate[1]

            # Convert column character to a 0-based index (a=0, b=1, ..., h=7)
            col_index = ord(col_char) - ord('a')

            # Convert row character to a 0-based index (1=0, 2=1, ..., 8=7)
            # Note: The problem uses 1-based indexing for rows in the input,
            # but 0-based is often easier for parity calculations.
            # Alternatively, using 1-based row index (1-8) also works,
            # as long as it's consistent. Let's use 0-based for consistency.
            row_index = int(row_char) - 1

            # Calculate the sum of the indices
            index_sum = col_index + row_index

            # Return the parity (0 for even sum, 1 for odd sum)
            return index_sum % 2

        # Get the color parity for both coordinates
        parity1 = get_color_parity(coordinate1)
        parity2 = get_color_parity(coordinate2)

        # The squares have the same color if their parities are the same
        return parity1 == parity2