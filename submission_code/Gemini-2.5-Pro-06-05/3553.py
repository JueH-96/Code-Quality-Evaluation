class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        """
        Determines if two chessboard squares have the same color.
        """

        # The color of a square on a chessboard is determined by the parity
        # of the sum of its coordinate indices. We can map the file (letter)
        # and rank (number) to 0-indexed integers to check this.
        #
        # Column ('file'): 'a'->0, 'b'->1, ..., 'h'->7
        # Row ('rank'):    '1'->0, '2'->1, ..., '8'->7
        #
        # A square has one color if (col_idx + row_idx) is even, and the
        # other color if it's odd. Two squares share a color if the parity
        # of their index sums is the same.

        # Calculate the sum of 0-indexed coordinates for the first square.
        col1 = ord(coordinate1[0]) - ord('a')
        row1 = int(coordinate1[1]) - 1
        sum1 = col1 + row1
        
        # Calculate the sum of 0-indexed coordinates for the second square.
        col2 = ord(coordinate2[0]) - ord('a')
        row2 = int(coordinate2[1]) - 1
        sum2 = col2 + row2
        
        # Two squares have the same color if their coordinate sums have the
        # same parity (i.e., both are even or both are odd).
        return (sum1 % 2) == (sum2 % 2)