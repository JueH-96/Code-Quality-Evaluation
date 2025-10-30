class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Extract column and row from coordinate1
        col1 = ord(coordinate1[0]) - 96
        row1 = int(coordinate1[1])
        
        # Extract column and row from coordinate2
        col2 = ord(coordinate2[0]) - 96
        row2 = int(coordinate2[1])
        
        # Determine if sums have the same parity
        return (col1 + row1) % 2 == (col2 + row2) % 2