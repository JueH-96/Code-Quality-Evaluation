class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert the letter part (column) to a number between 1 and 8
        col1 = ord(coordinate1[0]) - ord('a') + 1
        col2 = ord(coordinate2[0]) - ord('a') + 1
        
        # Convert the row part (digit) to an integer
        row1 = int(coordinate1[1])
        row2 = int(coordinate2[1])
        
        # Compute the color by taking (col + row) % 2
        color1 = (col1 + row1) % 2
        color2 = (col2 + row2) % 2
        
        # Return true if both squares have the same color
        return color1 == color2