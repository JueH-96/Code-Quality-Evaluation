class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert the letter to a number index (a=1, b=2, ..., h=8)
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # The square's color can be determined by the parity of the sum of its row and column indices.
        # Squares with the same parity are of the same color.
        return ((col1 + row1) % 2) == ((col2 + row2) % 2)