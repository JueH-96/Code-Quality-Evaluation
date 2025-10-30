class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        
        # Helper function to determine the "color parity" of a square.
        # On a chessboard, squares alternate colors. We can assign a numerical
        # value (0 or 1) to these colors based on the sum of their 0-indexed coordinates.
        #
        # 0-indexed coordinates:
        # Column: 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
        # Row:    '1' -> 0, '2' -> 1, ..., '8' -> 7
        #
        # A square (col_idx, row_idx) has one color if (col_idx + row_idx) is even,
        # and the other color if (col_idx + row_idx) is odd.
        # This function returns (col_idx + row_idx) % 2, which will be 0 for one
        # color and 1 for the other.
        def get_color_parity(coordinate: str) -> int:
            col_char = coordinate[0]
            row_char = coordinate[1]
            
            # Convert column character ('a'-'h') to 0-indexed integer (0-7)
            # e.g., ord('a') - ord('a') = 0
            #       ord('b') - ord('a') = 1
            col_idx = ord(col_char) - ord('a')
            
            # Convert row character ('1'-'8') to 0-indexed integer (0-7)
            # e.g., int('1') - 1 = 0
            #       int('2') - 1 = 1
            row_idx = int(row_char) - 1
            
            # The sum of 0-indexed coordinates determines the color.
            # Its parity (0 for even, 1 for odd) distinguishes the two colors.
            return (col_idx + row_idx) % 2

        # Get the color parity for the first coordinate
        parity1 = get_color_parity(coordinate1)
        
        # Get the color parity for the second coordinate
        parity2 = get_color_parity(coordinate2)
        
        # The squares have the same color if and only if their parities are the same.
        return parity1 == parity2