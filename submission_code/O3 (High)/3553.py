class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        """
        Returns True if the two given chessboard squares have the same color,
        and False otherwise.
        A square's color is determined by the parity of (column + row):
          - even  -> black  (e.g., 'a1')
          - odd   -> white
        """
        def color(coordinate: str) -> int:
            # Convert column letter to 1-based index: 'a' -> 1, 'b' -> 2, ..., 'h' -> 8
            col = ord(coordinate[0]) - ord('a') + 1
            # Row is already 1-based numeric character
            row = int(coordinate[1])
            # Color encoded as 0 (black) or 1 (white) by parity
            return (col + row) % 2
        
        # Squares have same color if their parity (color) values match
        return color(coordinate1) == color(coordinate2)