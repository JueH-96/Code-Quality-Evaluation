class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert the letter to a numeric column index (1-based)
        col1 = ord(coordinate1[0]) - ord('a') + 1
        col2 = ord(coordinate2[0]) - ord('a') + 1
        
        # Convert the row character to an integer
        row1 = int(coordinate1[1])
        row2 = int(coordinate2[1])
        
        # Determine the color by checking the parity of the sum of row and column
        color1 = (col1 + row1) % 2
        color2 = (col2 + row2) % 2
        
        # Return True if both squares are the same color, otherwise False
        return color1 == color2