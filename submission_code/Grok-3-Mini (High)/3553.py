class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert column letters to numbers (a=1, b=2, ..., h=8)
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # Check if the parity of the sum of coordinates is the same
        return (col1 + row1) % 2 == (col2 + row2) % 2