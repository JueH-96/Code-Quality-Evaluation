class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        # Convert column letter to number (a=1, b=2, ..., h=8)
        col1 = ord(coordinate1[0]) - ord('a') + 1
        row1 = int(coordinate1[1])
        
        col2 = ord(coordinate2[0]) - ord('a') + 1
        row2 = int(coordinate2[1])
        
        # Two squares have the same color if the sum of their coordinates have the same parity
        return (col1 + row1) % 2 == (col2 + row2) % 2